# Camera

### Common world coordinate system conventions
```
OpenGL/MAYA       OpenCV/Colmap     Blender      Unity/DirectX     Unreal   
Right-handed      Right-handed    Right-handed    Left-handed    Left-handed

     +y                +z           +z  +y         +y  +z          +z
     |                /             |  /           |  /             |
     |               /              | /            | /              |
     |______+x      /______+x       |/_____+x      |/_____+x        |______+x
    /               |                                              /
   /                |                                             / 
  /                 |                                            /      
 +z                 +y                                          +y
```
A common color code: x = <span style="color:red">red</span>, y = <span style="color:green">green</span>, z = <span style="color:blue">blue</span> (XYZ=RGB)

> Left/right-handed notation: 
roll your left/right palm from x to y, and your thumb should point to z.


### Camera pose matrix
```
[[Right_x, Up_x, Forward_x, Position_x],
 [Right_y, Up_y, Forward_y, Position_y],
 [Right_z, Up_z, Forward_z, Position_z],
 [0,       0,    0,         1         ]]
```

The xyz follows corresponding world coordinate system.
However, the three directions (right, up, forward) can be defined differently:
* forward can be `camera --> target` or `target --> camera`.
* up can align with the world-up-axis (`y`) or world-down-axis (`-y`).
* right can also be left, depending on it's (`up cross forward`) or (`forward cross up`).

### Common camera coordinate conventions
```
   OpenGL                OpenCV       
   Blender               Colmap       

     up  target          forward & target
     |  /                /        
     | /                /         
     |/_____right      /______right
    /                  |          
   /                   |          
  /                    |          
forward                up         
```

A common color code: right = <span style="color:red">red</span>., up = <span style="color:green">green</span>, forward = <span style="color:blue">blue</span> (XYZ=RUF=RGB).

### Our camera convention
* world coordinate is OpenGL/right-handed, `+x = right, +y = up, +z = forward`
* camera coordinate is OpenGL (forward is `target --> campos`).
* elevation in (-90, 90), from `+y (-90) --> -y (+90)`
* azimuth in (-180, 180), from `+z (0/-360) --> +x (90/-270) --> -z (180/-180) --> -x (270/-90) --> +z (360/0)`

### API

.. automodule:: kiui.cam
   :members: