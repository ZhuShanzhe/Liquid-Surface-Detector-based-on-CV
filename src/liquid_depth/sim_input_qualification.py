"""Simulation input eligibility, before any classifier or measurement.

Uses producer-only semantic visibility, not the estimator or its error. Never
passed to a meter; not a deployable real-camera observability oracle.
"""
import numpy as np

def qualify_simulated_input(depth_m,semantic,*,material,has_floats=False,
                            correlations=None,phase_roi=None,minimum_depth_m=0.):
    depth=np.asarray(depth_m);semantic=np.asarray(semantic)
    if depth.ndim!=2 or semantic.shape!=depth.shape:
        raise ValueError('aligned_simulation_visibility_required')
    reasons=[];audit=dict(simulation_only=True,used_for_measurement=False,
        uses_estimate_or_estimation_error=False,source='producer_semantic_visibility_or_raw_phase_modulation')
    if material!='ordinary_diffuse':
        if correlations is None or phase_roi is None:
            reasons.append('phase_support_missing')
        else:
            c=np.asarray(correlations);roi=np.asarray(phase_roi,bool)
            if c.ndim!=4 or c.shape[1]!=4 or c.shape[-2:]!=roi.shape or not roi.any():
                reasons.append('phase_support_missing')
            else:
                valid=np.isfinite(c).all(axis=(0,1))&(np.ptp(c,axis=1).max(axis=0)>1e-6)
                count=int((valid&roi).sum());invalid=float((roi&~valid).sum()/roi.sum())
                audit.update(phase_valid_pixels=count,phase_invalid_fraction=invalid)
                if count<30:reasons.append('insufficient_phase_support')
                if invalid>.5+1e-9:reasons.append('phase_invalid_fraction_above_50pct')
    else:
        valid=np.isfinite(depth)&(depth>0)&(depth>=minimum_depth_m)
        liquid=semantic==2;count=int((liquid&valid).sum());invalid=float((liquid&~valid).sum()/max(1,liquid.sum()))
        yy,xx=np.nonzero(liquid&valid);span=min(int(np.ptp(yy))+1,int(np.ptp(xx))+1) if len(xx) else 0
        audit.update(liquid_visible_pixels=int(liquid.sum()),liquid_valid_pixels=count,
                     liquid_invalid_fraction=invalid,minimum_axis_span_px=span)
        if count<30 or span<3:reasons.append('insufficient_visible_liquid_returns')
        if invalid>.5+1e-9:reasons.append('liquid_invalid_fraction_above_50pct')
        if has_floats:
            fy,fx=np.nonzero((semantic==3)&valid)
            fspan=min(int(np.ptp(fy))+1,int(np.ptp(fx))+1) if len(fx) else 0
            audit.update(float_valid_pixels=len(fx),float_minimum_axis_span_px=fspan)
            if len(fx)<3 or fspan<2:reasons.append('insufficient_float_support')
    return dict(eligible=not reasons,reasons=reasons,**audit)

def measure_if_eligible(meter,observation,qualification):
    """The only model call is inside the successful eligibility branch."""
    if not qualification['eligible']:return None
    return meter.measure(observation)
