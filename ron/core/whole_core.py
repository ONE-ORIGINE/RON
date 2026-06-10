from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from .quaternion import qexp, qmul, qnorm, qrotate
from .cube_memory import write_oriented_cube
from .moments import cube_moments, active_moment_ops, moment_exponents
from .signatures import update_signature, signature_vector3, signature_size
from .harmonics import real_harmonics, harmonic_axis
from .ports import port_projection, boundary_from_ports, face_normals
from .routing import neighbor_index, route_energy
from .tri_residual import tri_residual
from .causal_tree import causal_tree
from .symphony import conductor, entropy6
from .sphere_cube import prox_cube
from ron.codecs.vision_codex import RONVisionCodex

@dataclass
class RONWholeOutput:
    main: Tensor
    past: Tensor
    future: Tensor
    present: Tensor
    cause: Tensor
    reverse: Tensor
    consequence: Tensor
    diagonal: Tensor
    residual: Tensor
    q: Tensor
    omega: Tensor
    energy: Tensor
    moments: Tensor
    signature: Tensor
    ports: Tensor
    diagnostics: dict

class RONWholeTriAxialCore(nn.Module):
    """RON V20 Whole RON core.

    RON V20 keeps the operational RON core and adds stronger RON-native plasticity hooks: the residual, Phi,
    moments, signature, harmonics, ports, routing, causal tree and release are
    not separate decorations. They are coupled in the motion loop and audited by
    explicit operator-flow diagnostics.
    """
    def __init__(self, vocab:int=12, grid:int=2, cube:int=5, moment_degree:int=3, signature_degree:int=3, harmonic_degree:int=3, internal_cycles:int=3, seed:int=123):
        super().__init__()
        self.vocab=vocab; self.grid=grid; self.n_ron=grid**3; self.cube=cube
        self.moment_degree=moment_degree; self.signature_degree=signature_degree; self.harmonic_degree=harmonic_degree; self.internal_cycles=internal_cycles
        k=sum((d+1)*(d+2)//2 for d in range(1,moment_degree+1))

        # Geometric gains only; no dense classifier.
        self.axis_gain=nn.Parameter(torch.tensor([0.92,1.04,0.88]))
        self.event_energy_gain=nn.Parameter(torch.tensor(0.52))
        self.metric_weight=nn.Parameter(torch.ones(k)*0.32)
        self.impulse_weight=nn.Parameter(torch.zeros(k))
        self.port_bias=nn.Parameter(torch.zeros(6))
        self.routing_gain=nn.Parameter(torch.tensor(0.42))
        self.residual_to_port=nn.Parameter(torch.tensor(1.10))
        self.residual_to_motion=nn.Parameter(torch.tensor(0.62))
        self.phi_to_port=nn.Parameter(torch.tensor(0.34))
        self.phi_release_gain=nn.Parameter(torch.tensor(0.42))
        self.consequence_gain=nn.Parameter(torch.tensor(0.62))
        self.signature_gain=nn.Parameter(torch.tensor(0.45))
        self.harmonic_gain=nn.Parameter(torch.tensor(0.30))
        self.release_threshold=nn.Parameter(torch.tensor(0.24))
        self.causal_gain=nn.Parameter(torch.ones(6)*0.35)
        self.plasticity_gain=nn.Parameter(torch.tensor(0.10))
        self.register_buffer('loop_confidence_gain', torch.tensor(0.38), persistent=False)
        self.register_buffer('operator_balance_gain', torch.tensor(0.08), persistent=False)

        self.vision=RONVisionCodex(max_events=16)
        self.register_buffer('token_phase', self._phase(vocab,seed), persistent=False)
        self.register_buffer('ron_offsets', self._ron_offsets(grid), persistent=False)
        self.register_buffer('neigh', neighbor_index(grid), persistent=False)

    def project_parameters_(self):
        """RON-Prox projection after optimizer steps."""
        with torch.no_grad():
            self.routing_gain.clamp_(0.02, 2.0)
            self.residual_to_port.clamp_(0.02, 3.0)
            self.residual_to_motion.clamp_(0.0, 2.5)
            self.phi_to_port.clamp_(0.0, 2.0)
            self.phi_release_gain.clamp_(0.0, 2.0)
            self.event_energy_gain.clamp_(0.02, 2.0)
            self.consequence_gain.clamp_(0.0, 2.0)
            self.signature_gain.clamp_(0.0, 2.0)
            self.harmonic_gain.clamp_(0.0, 2.0)
            self.release_threshold.clamp_(0.02, 2.0)
            self.plasticity_gain.clamp_(0.0, 1.0)


    def ron_adapt_(self, signal):
        """RON-native local plasticity after a gradient-assisted step.

        This is deliberately small and bounded. It does not create a dense head;
        it changes the RON's own gains according to its geometric state:
        - high dissonance/residual increases residual-to-motion coupling;
        - low port entropy increases port diversity pressure;
        - excessive release raises the release threshold;
        - high operator imbalance nudges weak moment/signature/harmonic gains.
        """
        with torch.no_grad():
            residual = float(getattr(signal, "residual", 0.0))
            phi = float(getattr(signal, "phi_pressure", 0.0))
            diss = float(getattr(signal, "dissonance", 0.0))
            energy = float(getattr(signal, "energy", 0.0))
            port_entropy = float(getattr(signal, "port_entropy", 1.0))
            release = float(getattr(signal, "release", 0.0))
            op_balance = float(getattr(signal, "operator_balance", 0.0))

            tension = max(0.0, min(1.0, 0.40*residual + 0.30*diss + 0.20*phi + 0.10*(1.0-port_entropy)))
            self.residual_to_motion.add_(0.0025 * tension)
            self.residual_to_port.add_(0.0020 * tension)
            self.phi_to_port.add_(0.0015 * max(0.0, phi))
            self.phi_release_gain.add_(0.0010 * max(0.0, phi))

            if port_entropy < 0.55:
                # Slightly spread port bias back toward zero.
                self.port_bias.mul_(0.995)
            if release > 0.80 and residual > 0.05:
                self.release_threshold.add_(0.0015)
            elif release < 0.25 and residual < 0.04:
                self.release_threshold.sub_(0.0010)

            if op_balance > 1.5:
                # Strengthen underused deep operators, bounded by project_parameters_.
                self.signature_gain.add_(0.0010)
                self.harmonic_gain.add_(0.0008)
                self.plasticity_gain.add_(0.0006)
                # Moment influence comes from active torque, so keep motion coupling alive.
                self.residual_to_motion.add_(0.0005)

            if energy > 1.20:
                self.event_energy_gain.mul_(0.998)
            self.project_parameters_()

    def _phase(self,vocab,seed):
        ids=torch.arange(vocab,dtype=torch.float32)
        return torch.stack([
            torch.sin(2*math.pi*(ids+1)/(vocab+1)),
            torch.cos(2*math.pi*(2*ids+1)/(vocab+3)),
            torch.sin(2*math.pi*(3*ids+2)/(vocab+5))
        ],-1)

    def _ron_offsets(self,grid):
        vals=torch.linspace(-0.18,0.18,grid)
        z,y,x=torch.meshgrid(vals,vals,vals,indexing='ij')
        return torch.stack([x,y,z],-1).reshape(-1,3)

    def token_axes(self,tokens:Tensor):
        phase=self.token_phase.to(tokens.device)[tokens.long().clamp(0,self.vocab-1)]
        shifts=torch.tensor([-0.18,0.0,0.18],device=tokens.device,dtype=phase.dtype).view(1,3,1)
        axes=torch.tanh((phase + shifts*torch.roll(phase,1,-1))*self.axis_gain.view(1,1,3))
        axes=axes/axes.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        energy=(0.18+self.event_energy_gain.abs()*(0.5+0.5*axes.abs().mean(-1))).clamp_max(2.0)
        return axes, energy

    def _apply_mask(self, axes: Tensor, energies: Tensor, mask: Tensor|None):
        if mask is None:
            return axes, energies
        # Unknown axes become low-energy neutral axes, forcing inference through remaining axes.
        neutral=torch.tensor([0.0,0.0,1.0],device=axes.device,dtype=axes.dtype).view(1,1,3)
        known=axes.clone()
        e=energies.clone()
        def hide(index, cond):
            known[:,index,:]=torch.where(cond.view(-1,1), neutral.expand(axes.shape[0],1,3).squeeze(1), known[:,index,:])
            e[:,index]=torch.where(cond, e[:,index]*0.12, e[:,index])
        hide(0, (mask==1)|(mask==4)|(mask==6))
        hide(1, (mask==2)|(mask==4)|(mask==5))
        hide(2, (mask==3)|(mask==5)|(mask==6))
        return known, e

    def _state_from_axis(self, axis:Tensor, energy:Tensor, prev_sig:Tensor|None=None):
        B=axis.shape[0]; R=self.n_ron; dev=axis.device; dtype=axis.dtype
        offsets=self.ron_offsets.to(dev,dtype).view(1,R,3)
        omega=torch.tanh(axis[:,None,:]+offsets)
        q=qexp(omega*energy[:,None,None])
        cube,grid=write_oriented_cube(q, energy[:,None].expand(B,R), self.cube)
        moments,exps=cube_moments(cube,grid,self.moment_degree)
        ops=active_moment_ops(moments,exps)
        sig=torch.zeros(B,R,signature_size(self.signature_degree),device=dev,dtype=dtype) if prev_sig is None else prev_sig
        sig=update_signature(sig, omega+0.18*ops['torque'], self.signature_degree)
        sig_axis=signature_vector3(sig,self.signature_degree)
        h=real_harmonics(qrotate(q, torch.tensor([1.,0.,0.],device=dev,dtype=dtype))+0.25*ops['curvature']+0.15*sig_axis, self.harmonic_degree)
        h_axis=harmonic_axis(h)
        features=moments + 0.10*torch.nn.functional.pad(ops['torque'], (0,max(0,moments.shape[-1]-3)))[...,:moments.shape[-1]]
        return {'axis':axis,'energy':energy,'q':q,'omega':omega,'cube':cube,'moments':moments,'exps':exps,'ops':ops,'signature':sig,'sig_axis':sig_axis,'harmonics':h,'harm_axis':h_axis,'features':features}


    def _shared_field(self, q_axis: Tensor, torque: Tensor, sig_axis: Tensor, h_axis: Tensor, residual_axis: Tensor, phi_axis: Tensor, energy: Tensor, stability: Tensor|None=None):
        """Compose the RON field before ports.

        This is the anti-stacking rule: every operator must meet inside the same
        geometric field before the boundary decision. The returned diagnostics
        make it testable whether each operator contributes.
        """
        if stability is None:
            stability = torch.ones_like(energy)
        stab = stability.unsqueeze(-1).clamp(0.0, 1.0)
        energy_gate = torch.tanh(energy.unsqueeze(-1)).clamp(0.0, 1.0)
        c_q = q_axis
        c_tau = 0.46 * torque * (0.65 + 0.35*stab)
        c_sig = 0.34 * sig_axis
        c_h = 0.26 * h_axis
        c_res = 0.82 * residual_axis * (0.75 + 0.25*energy_gate)
        c_phi = -0.22 * phi_axis
        field = torch.tanh(c_q + c_tau + c_sig + c_h + c_res + c_phi)
        contrib = {
            'op_q': c_q.pow(2).mean(),
            'op_moment': c_tau.pow(2).mean(),
            'op_signature': c_sig.pow(2).mean(),
            'op_harmonic': c_h.pow(2).mean(),
            'op_residual': c_res.pow(2).mean(),
            'op_phi': c_phi.pow(2).mean(),
        }
        vals=torch.stack([contrib[k] for k in ['op_q','op_moment','op_signature','op_harmonic','op_residual','op_phi']])
        contrib['operator_balance'] = vals.std() / vals.mean().clamp_min(1e-6)
        return field, contrib

    def forward(self, triplet:Tensor|None=None, axes:Tensor|None=None, image:Tensor|None=None, depth:Tensor|None=None, prev_image:Tensor|None=None, mask:Tensor|None=None, adaptive_cycles:bool=False, no_phi:bool=False, no_residual:bool=False, no_bidirectional:bool=False, no_ports:bool=False, no_signature:bool=False, no_cube:bool=False, no_causal_tree:bool=False):
        if axes is None:
            if image is not None:
                axes, energies = self.vision.image_axes(image, depth, prev_image)
            elif triplet is not None:
                axes, energies = self.token_axes(triplet)
            else:
                raise ValueError('need triplet, axes or image')
        else:
            energies=torch.ones(axes.shape[:2],device=axes.device,dtype=axes.dtype)
        axes, energies = self._apply_mask(axes, energies, mask)

        X=self._state_from_axis(axes[:,0], energies[:,0])
        Y=self._state_from_axis(axes[:,1], energies[:,1])
        Z=self._state_from_axis(axes[:,2], energies[:,2])
        if no_cube:
            zero=torch.zeros_like(Y['omega'])
            X['features']=X['features']*0; Y['features']=Y['features']*0; Z['features']=Z['features']*0
            Y['ops']={'torque':zero,'curvature':zero,'stability':torch.ones_like(Y['energy'][:,None].expand(-1,self.n_ron))}

        tri=tri_residual(
            X['features'],Y['features'],Z['features'],
            X['axis'][:,None,:].expand(-1,self.n_ron,-1),
            Y['axis'][:,None,:].expand(-1,self.n_ron,-1),
            Z['axis'][:,None,:].expand(-1,self.n_ron,-1),
            self.metric_weight,self.impulse_weight,
            use_phi=not no_phi,bidirectional=not no_bidirectional
        )

        residual_axis=torch.zeros_like(tri.residual_axis) if no_residual else self.residual_to_port.abs()*tri.residual_axis
        residual_cost=torch.zeros_like(tri.cost) if no_residual else tri.cost
        phi_pressure=torch.zeros_like(tri.phi_pressure) if no_phi else tri.phi_pressure
        phi_axis=torch.zeros_like(tri.boundary_axis) if no_phi else self.phi_to_port.abs()*tri.boundary_axis

        sig_axis=torch.zeros_like(Y['sig_axis']) if no_signature else Y['sig_axis']
        q=Y['q']; omega=Y['omega']; moments=Y['moments']; sig=Y['signature']; ops=Y['ops']; h_axis=Y['harm_axis']
        routed_energy=energies[:,1,None].expand(-1,self.n_ron).clone()

        raw_total=torch.zeros(axes.shape[0],6,device=axes.device,dtype=axes.dtype)
        cycles=0
        loop_energy_moved=[]
        loop_q_delta=[]
        loop_release=[]
        loop_continue=[]

        # Pre-cycle shared field. Phi and residual already affect port access.
        q_axis_y=qrotate(q, torch.tensor([1.,0.,0.],device=axes.device,dtype=axes.dtype))
        shared_axis, contrib = self._shared_field(q_axis_y, ops['torque'], sig_axis, h_axis, residual_axis, phi_axis, routed_energy, ops.get('stability'))
        ports=port_projection(shared_axis, routed_energy, ops['torque'], sig_axis, h_axis, residual_axis-0.25*phi_axis, self.port_bias)
        if no_ports:
            ports=torch.ones_like(ports)/6.0

        for c in range(self.internal_cycles):
            # Route current energy first.
            routed_energy,raw=route_energy(routed_energy, ports, self.neigh.to(axes.device), self.routing_gain.abs(), absorb=0.28)
            raw_total=raw_total+raw

            # Main V14 improvement: residual and Phi pressure modify motion before
            # q, cube, moments, signature and ports are recomputed.
            moment_drive = 0.38*ops['torque']*(0.65+0.35*ops.get('stability', torch.ones_like(routed_energy)).unsqueeze(-1))
            residual_drive = self.residual_to_motion.abs()*0.48*residual_axis
            phi_drive = -0.22*phi_axis
            signature_drive = self.signature_gain.abs()*0.12*sig_axis
            harmonic_drive = self.harmonic_gain.abs()*0.10*h_axis
            motion_drive = moment_drive + residual_drive + phi_drive + signature_drive + harmonic_drive
            phi_damp = (1.0 + self.phi_release_gain.abs()*phi_pressure[:,None,None]).clamp(1.0, 4.0)
            omega_prev = omega
            q_prev = q
            omega=(0.80*omega + motion_drive) / phi_damp
            omega=omega.clamp(-2.5,2.5)
            q=qnorm(qmul(q,qexp(omega)))
            loop_q_delta.append((q - q_prev).pow(2).mean())
            loop_energy_moved.append(raw.abs().mean())

            # Rewrite cube after residual-informed motion.
            cube,grid=write_oriented_cube(q, routed_energy.clamp_min(1e-4), self.cube)
            cube=prox_cube(cube, limit=2.20)
            moments,exps=cube_moments(cube,grid,self.moment_degree)
            ops=active_moment_ops(moments,exps)

            sig=update_signature(sig, omega+0.18*residual_axis+0.12*ops['torque']-0.06*phi_axis, self.signature_degree)
            sig_axis=torch.zeros_like(sig_axis) if no_signature else signature_vector3(sig,self.signature_degree)
            h=real_harmonics(qrotate(q, torch.tensor([1.,0.,0.],device=axes.device,dtype=axes.dtype))+0.20*ops['curvature']+0.10*sig_axis-0.08*phi_axis,self.harmonic_degree)
            h_axis=harmonic_axis(h)

            q_axis=qrotate(q, torch.tensor([1.,0.,0.],device=axes.device,dtype=axes.dtype))
            shared_axis, contrib = self._shared_field(q_axis, ops['torque'], sig_axis, h_axis, residual_axis, phi_axis, routed_energy, ops.get('stability'))
            ports=port_projection(shared_axis, routed_energy, ops['torque'], sig_axis, h_axis, residual_axis-0.25*phi_axis, self.port_bias)
            if no_ports:
                ports=torch.ones_like(ports)/6.0

            portH=entropy6(ports.mean(1))
            probe=conductor(raw_total+boundary_from_ports(ports,routed_energy,self.grid), residual_cost + self.phi_release_gain.abs()*phi_pressure.mean(-1), routed_energy.mean(1), ports.mean(1), portH, self.release_threshold)
            loop_release.append(probe['release'].mean())
            loop_continue.append(probe['continue_pressure'].mean())
            cycles += 1
            if adaptive_cycles and c>=1:
                boundary_probe = raw_total+boundary_from_ports(ports,routed_energy,self.grid)
                top = boundary_probe.topk(2,dim=-1).values
                margin = top[:,0]-top[:,1]
                ready = (probe['release'] > (0.50 + 0.10*self.loop_confidence_gain.abs())) & (probe['dissonance'] < 0.55) & (residual_cost < 0.65) & (margin > 0.05)
                if bool(ready.float().mean().detach().cpu() > 0.55):
                    break

        base_boundary=raw_total+boundary_from_ports(ports,routed_energy,self.grid)
        n=face_normals(axes.device, axes.dtype)
        q_axis=qrotate(q, torch.tensor([1.,0.,0.],device=axes.device,dtype=axes.dtype))
        shared_axis, final_contrib = self._shared_field(q_axis, ops['torque'], sig_axis, h_axis, residual_axis, phi_axis, routed_energy, ops.get('stability'))

        # Rich fields. Direct axes only bias the integrated RON field.
        past_field=torch.tanh(-shared_axis - 0.40*residual_axis + 0.18*X['axis'][:,None,:] - 0.07*phi_axis)
        future_field=torch.tanh(shared_axis + 0.46*residual_axis + 0.18*Z['axis'][:,None,:] - 0.07*phi_axis)
        present_field=torch.tanh(shared_axis + 0.18*Y['axis'][:,None,:] - 0.05*phi_axis)
        cause_field=torch.tanh(residual_axis + 0.35*ops['torque'] + 0.22*sig_axis - 0.10*phi_axis)
        reverse_field=torch.tanh(past_field - future_field + 0.18*sig_axis)
        consequence_field=torch.tanh(future_field + 0.40*cause_field - 0.25*past_field + 0.20*h_axis - 0.10*phi_axis)
        diagonal_field=torch.tanh(present_field + future_field - past_field + 0.35*cause_field - 0.20*reverse_field)

        past=torch.einsum('brd,fd->bf', past_field, n)+0.28*base_boundary
        future=torch.einsum('brd,fd->bf', future_field, n)+0.28*base_boundary
        present=torch.einsum('brd,fd->bf', present_field, n)+0.24*base_boundary
        cause=torch.einsum('brd,fd->bf', cause_field, n)+0.22*base_boundary
        reverse=torch.einsum('brd,fd->bf', reverse_field, n)+0.22*base_boundary
        consequence=torch.einsum('brd,fd->bf', consequence_field, n)+0.32*base_boundary
        diagonal=torch.einsum('brd,fd->bf', diagonal_field, n)+0.30*base_boundary
        main=0.42*future+0.24*diagonal+0.22*consequence+0.18*cause+0.12*base_boundary

        portH=entropy6(ports.mean(1))
        total_residual_pressure = residual_cost + self.phi_release_gain.abs()*phi_pressure.mean(-1)
        preliminary=conductor(main, total_residual_pressure, routed_energy.mean(1), ports.mean(1), portH, self.release_threshold)
        tree=causal_tree(main,past,future,cause,consequence,total_residual_pressure,preliminary['harmony'],preliminary['dissonance'],self.causal_gain) if not no_causal_tree else None
        if tree is not None:
            main=main+self.consequence_gain.abs()*tree.filtered
            access=tree.access; pressure=tree.pressure
        else:
            access=torch.sigmoid(main); pressure=total_residual_pressure

        cond=conductor(main, total_residual_pressure, routed_energy.mean(1), access, portH, self.release_threshold)
        main=cond['release'][:,None]*main+(1-cond['release'])[:,None]*(0.25*diagonal+0.15*base_boundary)

        # Lightweight plasticity diagnostic: not dynamic allocation yet.
        plasticity = torch.sigmoid(self.plasticity_gain.abs()*(routed_energy.mean(1)+cond['dissonance']-cond['harmony']))

        loop_energy = torch.stack(loop_energy_moved).mean() if loop_energy_moved else torch.zeros((),device=axes.device,dtype=axes.dtype)
        loop_delta = torch.stack(loop_q_delta).mean() if loop_q_delta else torch.zeros((),device=axes.device,dtype=axes.dtype)
        loop_rel = torch.stack(loop_release).mean() if loop_release else cond['release'].mean()
        loop_cont = torch.stack(loop_continue).mean() if loop_continue else torch.zeros((),device=axes.device,dtype=axes.dtype)
        diag={
            'residual_cost':residual_cost.mean(),
            'phi_pressure':phi_pressure.mean(),
            'barrier':tri.barrier,
            'harmony':cond['harmony'].mean(),
            'dissonance':cond['dissonance'].mean(),
            'branch_access':access.mean(),
            'release':cond['release'].mean(),
            'energy_mean':routed_energy.mean(),
            'port_entropy':portH.mean(),
            'cycles_used':torch.tensor(float(cycles),device=axes.device,dtype=axes.dtype),
            'consequence_pressure':pressure.mean(),
            'plasticity_pressure':plasticity.mean(),
            'loop_energy_moved':loop_energy,
            'loop_q_delta':loop_delta,
            'loop_release':loop_rel,
            'loop_continue':loop_cont,
            **final_contrib,
        }
        return RONWholeOutput(main,past,future,present,cause,reverse,consequence,diagonal,tri.residual,q,omega,routed_energy,moments,sig,ports,diag)
