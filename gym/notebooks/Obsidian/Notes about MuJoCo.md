
#### MuJoCo (Multi-Joint dynamics with Contact) Model Definition.
##### Key Concepts:
- **worldbody**: The root of the kinematic tree.
- **body**: A physical object (can have mass, geometry, etc.).
- **joint**: Allows motion between a body and its parent.
- **geom**: The visual and collision shape of a body.
- **actuator**: Defines how we control the joints (motors).

##### Simulation Settings
```xml
<compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
```
- `angle="radian"`: Interpretation of angle values (radian vs degree).
- `coordinate="local"`: Coordinate frame for defining bodies/geoms (local vs global).
- `inertiafromgeom="true"`: Automatically infer mass and inertia from geometry density and volume.
```xml
<option timestep="0.01" gravity="0 0 -9.81" density="1.2" viscosity="0.00002" integrator="RK4"/>
```
- timestep="0.01": Physics integration step size in seconds (10ms).
- gravity="0 0 -9.81": Gravity vector (x, y, z) in m/s^2.
- density="1.2": fluid density (e.g. air) for computing drag/lift forces.
- viscosity="0.00002": fluid viscosity for computing drag/lift forces.
- integrator="RK4": Numerical integrator (Runge-Kutta 4th order) for higher accuracy.

##### Default Settings for Classes of Elements
```xml
<default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
</default>
```
- `joint: limited="true"`: means joints have limits by default
- `geom: conaffinity="1"`: means these geoms can collide

##### Assets: Textures and Materials Used for Rendering
```xml
<asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
</asset>
```