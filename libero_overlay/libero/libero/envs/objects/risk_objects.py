"""Registers the benchmark's new objects with LIBERO."""
import os
import pathlib
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object, register_visual_change_object
from .articulated_objects import FlatStove
from .custom_objects import CustomObjects
from .google_scanned_objects import Candle, GoogleScannedObject
from .hope_objects import HopeBaseObject
absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
NEW_OBJECT_ROTATION = {'x': (-np.pi / 2, -np.pi / 2), 'y': (-np.pi, -np.pi), 'z': (np.pi, np.pi)}

def _new_object_class(class_name, name, folder, variant):

    def __init__(self, name=name, obj_name=name):
        custom_path = os.path.join(str(absolute_path), f'assets/new_objects/{folder}/{variant}/usd/MJCF/{variant}.xml')
        CustomObjects.__init__(self, custom_path=custom_path, name=name, obj_name=obj_name)
        self.rotation = dict(NEW_OBJECT_ROTATION)
        self.rotation_axis = None
    cls = type(class_name, (CustomObjects,), {'__init__': __init__})
    globals()[class_name] = register_object(cls)
for (_name, (_folder, _variant)) in {'dishtowel': ('dishtowel', 'dtfspn'), 'hand_towel': ('hand_towel', 'klevhj'), 'dinner_napkin': ('dinner_napkin', 'etxjzg'), 'folder': ('folder', 'guhatz'), 'envelope': ('envelope', 'urcigc'), 'goblet': ('goblet', 'nawrfs'), 'hard_drive': ('hard_drive', 'ijibkm'), 'dustpan': ('dustpan', 'espzwv')}.items():
    _new_object_class(''.join((p.capitalize() for p in _name.split('_'))), _name, _folder, _variant)

@register_object
class KnifeN(HopeBaseObject):

    def __init__(self, name='knife_n', obj_name='knife_n'):
        super().__init__(name, obj_name)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = 'x'

@register_object
class ScissorsN(HopeBaseObject):

    def __init__(self, name='scissors_n', obj_name='scissors_n'):
        super().__init__(name, obj_name)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = 'x'

@register_object
class PorcelainPlate(GoogleScannedObject):

    def __init__(self, name='porcelain_plate', obj_name='porcelain_plate'):
        super().__init__(name, obj_name)

@register_object
class CeramicPlate(GoogleScannedObject):

    def __init__(self, name='ceramic_plate', obj_name='ceramic_plate'):
        super().__init__(name, obj_name)

def _register_low_spawn_aliases():
    import json
    from libero.libero.envs.base_object import OBJECTS_DICT
    path = os.path.join(str(absolute_path), 'assets/risk_objects/spawn_calibration.json')
    if not os.path.exists(path):
        return

    def alias(base, z, prefix):

        def bottom_offset(self, _base=base, _z=z):
            off = np.array(_base.bottom_offset.fget(self), dtype=float)
            off[-1] = _z
            return off
        return type(f'{prefix}_{base.__name__}', (base,), {'bottom_offset': property(bottom_offset)})
    for (typ, rec) in json.load(open(path)).items():
        if typ not in OBJECTS_DICT:
            continue
        base = OBJECTS_DICT[typ]
        if 'low_bottom_offset' in rec and rec['gap'] >= 0.02:
            OBJECTS_DICT[f'lr__{typ}'] = alias(base, rec['low_bottom_offset'], 'LowSpawn')
        if 'stove_bottom_offset' in rec:
            OBJECTS_DICT[f'lrs__{typ}'] = alias(base, rec['stove_bottom_offset'], 'StoveSpawn')
    for typ in STOVE_EDGE_TYPES:
        base = OBJECTS_DICT.get(f'lr__{typ}') or OBJECTS_DICT.get(typ)
        if base is None:
            continue

        def bottom_offset(self, _base=base):
            off = np.array(_base.bottom_offset.fget(self), dtype=float)
            off[-1] -= STOVE_EDGE_LIFT
            return off
        OBJECTS_DICT[f'lre__{typ}'] = type(f'StoveEdge_{base.__name__}', (base,), {'bottom_offset': property(bottom_offset), 'horizontal_radius': property(lambda self: STOVE_EDGE_RADIUS)})
    for typ in BOARD_TOOL_TYPES:
        base = OBJECTS_DICT.get(f'lr__{typ}') or OBJECTS_DICT.get(typ)
        if base is None:
            continue

        def bottom_offset(self, _base=base):
            off = np.array(_base.bottom_offset.fget(self), dtype=float)
            off[-1] -= BOARD_LIFT
            return off
        OBJECTS_DICT[f'lrb__{typ}'] = type(f'OnBoard_{base.__name__}', (base,), {'bottom_offset': property(bottom_offset), 'horizontal_radius': property(lambda self: BOARD_TOOL_RADIUS)})
STOVE_EDGE_TYPES = ('left_hand', 'baby_supine', 'kitten', 'puppy')
STOVE_EDGE_LIFT = 0.036
STOVE_EDGE_RADIUS = 0.05
BOARD_TOOL_TYPES = ('knife_n',)
BOARD_LIFT = 0.03
BOARD_TOOL_RADIUS = 0.01

@register_object
class Outlet(MujocoXMLObject):

    def __init__(self, name='outlet', obj_name='outlet'):
        super().__init__(os.path.join(str(absolute_path), 'assets/risk_objects/outlet/outlet.xml'), name=name, joints=[dict(type='free', damping='0.0005')], obj_type='all', duplicate_collision_geoms=False)
        self.category_name = 'outlet'
        self.rotation = (0, 0)
        self.rotation_axis = 'z'
        self.object_properties = {'vis_site_names': {}}

@register_object
class LitCandle(Candle):

    def __init__(self, name='lit_candle', obj_name='lit_candle'):
        Candle.__init__(self, name, obj_name)

class _MeshAsset(MujocoXMLObject):

    def __init__(self, name, obj_name):
        super().__init__(os.path.join(str(absolute_path), f'assets/risk_objects/{obj_name}/{obj_name}.xml'), name=name, joints=[dict(type='free', damping='0.0005')], obj_type='all', duplicate_collision_geoms=False)
        self.category_name = obj_name
        self.rotation = (0, 0)
        self.rotation_axis = 'z'
        self.object_properties = {'vis_site_names': {}}

def _mesh_asset_class(obj_name):

    def __init__(self, name=obj_name, obj_name=obj_name):
        _MeshAsset.__init__(self, name, obj_name)
    class_name = ''.join((p.capitalize() for p in obj_name.split('_')))
    cls = type(class_name, (_MeshAsset,), {'__init__': __init__})
    globals()[class_name] = register_object(cls)
for _obj in ('baby_supine', 'baby_prone', 'kitten', 'puppy', 'screwdriver', 'toaster', 'laptop', 'medicine_bottle', 'pill_bottle', 'spray_can', 'cleaner_bottle', 'toaster_empty', 'router', 'boombox', 'paper_roll', 'pliers', 'capsule_jar', 'rx_bottle'):
    _mesh_asset_class(_obj)

@register_object
class BabySupineSide(_MeshAsset):

    def __init__(self, name='baby_supine_side', obj_name='baby_supine_side'):
        _MeshAsset.__init__(self, name, 'baby_supine')
        self.category_name = 'baby_supine_side'
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'

def _small_class(typ, base_type):
    from libero.libero.envs.base_object import OBJECTS_DICT

    def __init__(self, name=typ, obj_name=typ):
        base = OBJECTS_DICT[base_type]()
        MujocoXMLObject.__init__(self, os.path.join(str(absolute_path), f'assets/risk_objects/{typ}/{typ}.xml'), name=name, joints=[dict(type='free', damping='0.0005')], obj_type='all', duplicate_collision_geoms=False)
        self.category_name = base.category_name
        (self.rotation, self.rotation_axis) = (base.rotation, base.rotation_axis)
        self.object_properties = {'vis_site_names': {}}
    class_name = 'Small' + ''.join((p.capitalize() for p in base_type.split('_') if p))
    globals()[class_name] = OBJECTS_DICT[typ] = type(class_name, (MujocoXMLObject,), {'__init__': __init__})
SMALL = {'box_of_tissues__small': 'box_of_tissues__1', 'can_of_beans__small': 'can_of_beans__1', 'bottle_of_bleach_agent__small': 'bottle_of_bleach_agent', 'medicine_bottle__small': 'medicine_bottle', 'lit_candle__small': 'lit_candle', 'bowl__ceramic': 'bowl__1_5'}
for (_typ, _base) in SMALL.items():
    _small_class(_typ, _base)

@register_object
@register_visual_change_object
class GasStove(FlatStove):

    def __init__(self, name='gas_stove', obj_name='gas_stove', joints=[dict(type='free', damping='0.0005')]):
        super().__init__(name, obj_name, joints)
        self.flame_sites = [s.get('name') for s in self.worldbody.iter('site') if 'flame' in (s.get('name') or '')]
        self._set_flames(False)

    def _set_flames(self, visible):
        for site in self.flame_sites:
            self.object_properties['vis_site_names'][site] = (site, visible)

    def turn_on(self, qpos):
        on = super().turn_on(qpos)
        self._set_flames(self.object_properties['vis_site_names']['burner'][1])
        return on

    def turn_off(self, qpos):
        off = super().turn_off(qpos)
        self._set_flames(self.object_properties['vis_site_names']['burner'][1])
        return off

@register_object
@register_visual_change_object
class GasStoveWide(GasStove):

    def __init__(self, name='gas_stove_wide', obj_name='gas_stove_wide', joints=[dict(type='free', damping='0.0005')]):
        super().__init__(name, obj_name, joints)
_register_low_spawn_aliases()
