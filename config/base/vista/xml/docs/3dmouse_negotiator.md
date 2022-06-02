# Negotiator between mouse and controller position
```mermaid
classDiagram

class debug_text{
  <<SimpleText>>
}

class get_mouse_xpos{
  <<GetVariable[int]>>
  variable mouse_xpos
}

class get_mouse_ypos{
  <<GetVariable[int]>>
  variable mouse_ypos
}

class get_gamepad_xpos{
  <<GetVariable[int]>>
  variable gamepad_xpos
}

class get_gamepad_ypos{
  <<GetVariable[int]>>
  variable gamepad_ypos
}

class mouse_xpos_to_float{
  <<TypeConvert[int,float]>>
}

class mouse_ypos_to_float{
  <<TypeConvert[int,float]>>
}

class const_0{
  <<ConstantValue[float]>>
  value 0
}

class gamepad_xpos_to_float{
  <<TypeConvert[int,float]>>
}

class gamepad_ypos_to_float{
  <<TypeConvert[int,float]>>
}

class compose_mouse{
  <<Compose3DVector>>
}

class compose_gamepad{
  <<Compose3DVector>>
}

class cd_mouse{
  <<ChangeDetect[VistaVector3D]>>
}

class cd_gamepad{
  <<ChangeDetect[VistaVector3D]>>
}

class latest_pos{
  <<LatestUpdate[VistaVector3D]>>
}

class decompose_pos{
  <<Decompose3DVector>>
}

class xpos_to_int{
  <<TypeConvert[float,int]>>
}

class ypos_to_int{
  <<TypeConvert[float,int]>>
}

class 3dmouse{
  <<3DMouseTransform>>
  displaysystem MAIN
  viewport MAIN_VIEWPORT
  in_world_coordinates FALSE
  origin_offset_along_ray 0
}

class 3dmouse_matrix{
  <<MatrixCompose>>
}

class intention_transform{
  <<SetTransform>>
  object SELECTION_NODE
}

get_mouse_xpos --> mouse_xpos_to_float : value --> in
get_mouse_ypos --> mouse_ypos_to_float : value --> in
get_gamepad_xpos --> gamepad_xpos_to_float : value --> in
get_gamepad_ypos --> gamepad_ypos_to_float : value --> in
mouse_xpos_to_float --> compose_mouse : out --> x
mouse_ypos_to_float --> compose_mouse : out --> y
const_0 --> compose_mouse : value --> z
const_0 --> compose_gamepad : value --> z
gamepad_xpos_to_float --> compose_gamepad : out --> x
gamepad_ypos_to_float --> compose_gamepad : out --> y
compose_mouse --> cd_mouse : out --> in
compose_gamepad --> cd_gamepad : out --> in
cd_mouse --> latest_pos : out --> mouse
cd_gamepad --> latest_pos : out --> gamepad
latest_pos --> decompose_pos : out --> in
decompose_pos --> xpos_to_int : x --> in
decompose_pos --> ypos_to_int : y --> in
xpos_to_int --> 3dmouse : out --> x_pos
ypos_to_int --> 3dmouse : out --> y_pos
3dmouse --> 3dmouse_matrix : orientation --> orientation
3dmouse --> 3dmouse_matrix : position --> translation
3dmouse_matrix --> intention_transform : out --> in

