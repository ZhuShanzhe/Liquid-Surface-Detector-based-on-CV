"""Explicit measurement-path boundaries, separate from lighting/occlusion tags.

This contract does not classify material physics from RGB brightness or missing
depth. New routes are opt-in; legacy profiles keep their previous behavior.
"""

def resolve_measurement_route(context):
    route = context.get('measurement_route')
    if route is None:
        return dict(route='legacy_unspecified',allowed=True,reasons=[],research_only=False)
    if route not in ('surface', 'homogeneous_refraction', 'multilayer'):
        raise ValueError('unknown measurement_route')
    optics=context.get('liquid_optics','unknown')
    if optics not in ('ordinary','transparent','translucent_mild','strong_scattering','unknown'):
        raise ValueError('unknown liquid_optics')
    reasons=[]
    if route=='surface':
        if context.get('model_variant') == 'transparent_multilayer':
            reasons.append('surface_route_conflicts_with_multilayer_model')
        if context.get('surface_return_verified') is not True:
            reasons.append('surface_return_not_verified')
        if context.get('multi_layer_expected'):
            reasons.append('multiple_interfaces_require_layer_selection')
    elif route=='homogeneous_refraction':
        if optics not in ('transparent','translucent_mild'):
            reasons.append('outside_homogeneous_transparent_scope')
        if context.get('multi_layer_expected') or context.get('glass_wall_in_optical_path'):
            reasons.append('multiple_interfaces_not_in_homogeneous_model')
        if context.get('bottom_pattern_visible') is not True:
            reasons.append('bottom_pattern_not_verified')
        # Research inverse modules are not industrial runtime model adapters.
        reasons.append('research_route_not_integrated')
    else:
        reasons.append('research_route_not_integrated')
    return dict(route=route,optics=optics,allowed=not reasons,reasons=reasons,
                research_only=route!='surface')
