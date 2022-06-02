## Controller Input
```mermaid
classDiagram
direction LR

class vrpn_analog_driver{
  <<DriverSensor>>
  driver XBOXCONTROLLER
  sensor 0
  type ANALOG
}

class vrpn_analog_data{
  <<HistoryProject>>
  project(VALUE, CHANNELS)
}

vrpn_analog_driver --> vrpn_analog_data : history --> history

class vrpn_button_driver{
  <<DriverSensor>>
  driver XBOXCONTROLLER
  sensor 0
  type BUTTON
}

class vrpn_button_data{
  <<HistoryProject>>
  project(BTMASK, NUMBER, STATE)
}

class btn_number_to_int{
  <<TypeConvert[unsigned int,int]>>
}

class get_button{
  <<Demultiplex[unsigned int]>>
  num_outports 14
}

vrpn_button_driver --> vrpn_button_data : history --> history
vrpn_button_data --> btn_number_to_int : NUMBER --> in
btn_number_to_int --> get_button : out --> select
vrpn_button_data --> get_button : STATE --> value
```
---
## Triggers & D-Pad to movement (panning & zooming)
```mermaid
classDiagram

class left_to_float{
  <<TypeConvert[unsigned int,float]>>
}

class right_to_float{
  <<TypeConvert[unsigned int,float]>>
}

class up_to_float{
  <<TypeConvert[unsigned int,float]>>
}

class down_to_float{
  <<TypeConvert_unsigned_int_float>>
}

class x_direction{
  <<Substract[float]>>
}

class const_pan_sensitivity{
  <<ConstantValue[float]>>
  value 1.0
}

class y_direction{
  <<Substract[float]>>
}

class x_movement{
  <<Multiply[float]>>
}

class y_movement{
  <<Multiply[float]>>
}

class triggers_to_float{
  <<TypeConvert[double,float]>>
}

class const_zoom_sensitivity{
  <<ConstantValue[float]>>
  value 1.0
}

class z_movement{
  <<Multiply[float]>>
}

class movement{
  <<Compose3DVector>>
}

get_button --> left_to_float : 13
get_button --> right_to_float : 11
get_button --> up_to_float : 10
get_button --> down_to_float : 12
right_to_float --> x_direction : out --> first
left_to_float --> x_direction : out --> second
down_to_float --> y_direction : out --> second
up_to_float --> y_direction : out --> first
x_direction --> x_movement : out --> first
const_pan_sensitivity --> x_movement : value --> second
y_direction --> y_movement : out --> first
const_pan_sensitivity --> y_movement : value --> second

vrpn_analog_data --> triggers_to_float : VALUE_4 --> in
triggers_to_float --> z_movement : out --> first
const_zoom_sensitivity --> z_movement : value --> second

z_movement --> movement : out --> z
x_movement --> movement : out --> x
y_movement --> movement : out --> y
```
---
## Right Stick & Shoulder Buttons to rotation (pitch, yaw, roll)
```mermaid
classDiagram

class const_pitch_sensitivity{
  <<ConstantValue[float]>>
  value 1.0
}
class pitch_RSy_to_float{
  <<TypeConvert[double,float]>>
}

class pitch_axis{
  <<ConstantValue[VistaVector3D]>>
  value -1,0,0,0
}

class pitch_angle{
  <<Multiply[float]>>
}

class pitch_rotation{
  <<AxisRotate>>
}

vrpn_analog_data --> pitch_RSy_to_float : VALUE_3 --> in
pitch_RSy_to_float --> pitch_angle : out --> first
const_pitch_sensitivity --> pitch_angle : value --> second
pitch_angle --> pitch_rotation : out --> angle
pitch_axis --> pitch_rotation : value --> axis

class const_yaw_sensitivity{
  <<ConstantValue[float]>>
  value 1.0
}

class yaw_RSx_to_float{
  <<TypeConvert[double,float]>>
}

class yaw_axis{
  <<ConstantValue[VistaVector3D]>>
  value 0,-1,0,0
}

class yaw_angle{
  <<Multiply[float]>>
}

class yaw_rotation{
  <<AxisRotate>>
}

vrpn_analog_data --> yaw_RSx_to_float : VALUE_2 --> in
yaw_RSx_to_float --> yaw_angle : out --> first
const_yaw_sensitivity --> yaw_angle : value --> second
yaw_angle --> yaw_rotation : out --> angle
yaw_axis --> yaw_rotation : value --> axis

class rb_to_float{
  <<TypeConvert[unsigned int,float]>>
}

class lb_to_float{
  <<TypeConvert[unsigned int,float]>>
}

class const_roll_sensitivity{
  <<ConstantType[float]>>
  value 1.0
}

class roll_direction{
  <<Substract[float]>>
}

class roll_angle{
  <<Multiply[float]>>
}

class roll_axis{
  <<ConstantValue[VistaVector3D]>>
  value 0,0,-1,0
}

class roll_rotation{
  <<AxisRotate>>
}

get_button --> rb_to_float : 4 --> in
get_button --> lb_to_float : 5 --> in
rb_to_float --> roll_direction : out --> second
lb_to_float --> roll_direction : out --> first
roll_direction --> roll_angle : out --> first
const_roll_sensitivity --> roll_angle : value --> second
roll_angle --> roll_rotation : out --> angle
roll_axis --> roll_rotation : value --> axis

class pitch_yaw{
  <<Multiply[VistaQuaternion]>>
}

class rotation{
  <<Multiply[VistaQuaternion]>>
}

pitch_rotation --> pitch_yaw : out --> first
yaw_rotation --> pitch_yaw : out --> second
pitch_yaw --> rotation : out --> first
roll_rotation --> rotation : out --> second
```
---
## navigation to observer
```mermaid
classDiagram

class observer{
  <<ObserverNavigationNode>>
  max_linear_speed 10, 10, 10
  max_angular_speed 1
}

class timer{
  <<Timer>>
}

movement --> observer : out --> translation
rotation --> observer : out --> rotation
timer --> observer : time --> time
```
---
## Buttons to Mouse Buttons
```mermaid
classDiagram

class A_to_bool{
  <<TypeConvert[unsigned int,bool]>>
}

class A_change_detect{
  <<ChangeDetect[bool]>>
}

class button_01{
  <<EventSource>>
  tag="button_01"
}

get_button --> A_to_bool : 0 --> in
A_to_bool --> A_change_detect : out --> in
A_change_detect --> button_01 : out --> value

class B_to_bool{
  <<TypeConvert[unsigned int,bool]>>
}

class B_change_detect{
  <<ChangeDetect[bool]>>
}

class button_02{
  <<EventSource>>
  tag="button_02"
}

get_button --> B_to_bool : 1 --> in
B_to_bool --> B_change_detect : out --> in
B_change_detect --> button_02 : out --> value

class X_to_bool{
  <<TypeConvert[unsigned int,bool]>>
}

class X_change_detect{
  <<ChangeDetect[bool]>>
}

class button_03{
  <<EventSource>>
  tag="button_03"
}

get_button --> X_to_bool : 2 --> in
X_to_bool --> X_change_detect : out --> in
X_change_detect --> button_03 : out --> value
```
---
## Left stick to mouse movement
```mermaid
classDiagram

class LSy_to_float{
  <<TypeConvert[double,float]>>
}

class const_mouse_sensitivity{
  <<ConstantValue[float]>>
  value 1.0
}

class LSx_to_float{
  <<TypeConvert[double,float]>>
}

class mouse_y_movement{
  <<Multiply[float]>>
}

class mouse_x_movement{
  <<Multiply[float]>>
}

class ymov{
  <<TypeConvert[float,int]>>
}

class xmov{
  <<TypeConvert[float,int]>>
}

vrpn_analog_data --> LSy_to_float : VALUE_1 --> in
vrpn_analog_data --> LSx_to_float : VALUE_0 --> in

LSy_to_float --> mouse_y_movement : out --> first
const_mouse_sensitivity --> mouse_y_movement : value --> second
LSx_to_float --> mouse_x_movement : out --> first
const_mouse_sensitivity --> mouse_x_movement : value --> second

mouse_y_movement --> ymov : out --> in
mouse_x_movement --> xmov : out --> in
```
---
## limit mouse position to viewport bounds
```mermaid
classDiagram

class add_ymov_ypos{
  <<Add[int]>>
}

class add_xmov_xpos{
  <<Add[int]>>
}

class lower_limit_ypos{
  <<Max[int]>>
}

class const_0{
  <<ConstantValue[int]>>
  value 0
}

class lower_limit_xpos{
  <<Max[int]>>
}

class upper_limit_ypos{
  <<Min[int]>>
}

class viewport{
  <<ViewportSource>>
  value MAIN_VIEWPORT
}

class upper_limit_xpos{
  <<Min[int]>>
}

class latest_ypos{
  <<LatestUpdate[int]>>
}

class latest_xpos{
  <<LatestUpdate[int]>>
}

class set_gamepad_ypos{
  <<SetVariable[int]>>
  variable gamepad_ypos
}

class set_gamepad_xpos{
  <<SetVariable[int]>>
  variable gamepad_xpos
}

ymov --> add_ymov_ypos : out --> first
xmov --> add_xmov_xpos : out --> first
add_ymov_ypos --> lower_limit_ypos : out --> first
add_xmov_xpos --> lower_limit_xpos : out --> first
const_0 --> lower_limit_ypos : value --> second
const_0 --> lower_limit_xpos : value --> second
lower_limit_ypos --> upper_limit_ypos : out --> first
lower_limit_xpos --> upper_limit_xpos : out --> first
viewport --> upper_limit_ypos : vp_h --> second
viewport --> upper_limit_xpos : vp_w --> second

upper_limit_ypos --> latest_ypos : out --> current
upper_limit_xpos --> latest_xpos : out --> current

latest_ypos --> add_ymov_ypos : out --> second
latest_xpos --> add_xmov_xpos : out --> second

latest_ypos --> set_gamepad_ypos : out --> value
latest_xpos --> set_gamepad_xpos : out --> value
```
---
## Set initial values for mouse X-Pos & Y-Pos
```mermaid
classDiagram

class const_2{
  <<ConstantValue[int]>>
  value 2
}

class get_viewport_center_y{
  <<Divide[int]>>
}

class get_viewport_center_x{
  <<Divide[int]>>
}

const_2 --> get_viewport_center_y : value --> second
const_2 --> get_viewport_center_x : value --> second
viewport --> get_viewport_center_y : vp_h --> first
viewport --> get_viewport_center_x : vp_w --> first
get_viewport_center_y --> latest_ypos : out --> 1
get_viewport_center_x --> latest_xpos : out --> 1
```
