# ForgeCAM AI Architecture

This application is a full CAM (Computer-Aided Manufacturing) pipeline. It does not simply "draw a line"; it calculates the physics and kinematics required to cut metal safely.

## The 5 Core Elements

1. **Geometry Engine:** Parses DXF/PDF/Images into strict mathematical lines.
2. **Tooling Model:** Defines the physical tool (Endmill diameter, Turning Insert corner radius). *Critical: A 6mm endmill cannot cut a 5mm internal corner.*
3. **Material Model:** Defines cutting speed (SFM) and feed factors for Aluminum, Steel, or Stainless.
4. **CAM Strategy Engine:** The "Brain". It calculates step-downs, offsets, and roughing passes.
5. **FANUC Postprocessor:** Translates the abstract path into FANUC 0i-MF (Mill) or 0i-TF (Lathe) syntax.

## How They Interact (The Flow)

### Scenario: Milling a Pocket
- **Geometry + Tooling:** The engine measures the pocket. It checks the `Tool.diameter`. If the tool is too big, the `ToolpathValidator` throws a warning to the frontend.
- **Material + Spindle:** The engine takes `Material.cutting_speed` and calculates `RPM = (Speed * 1000) / (PI * Tool.Diameter)`. It checks this against `Spindle_Limit`. If the RPM is too high, it caps it and warns the user.
- **Strategy:** It generates a zig-zag path, stepping down by `Depth_of_Cut` per pass.
- **Postprocessor:** It adds `G41` (Cutter Compensation) so the CNC machine knows to offset the tool by its own radius.

### Scenario: Turning a Shaft
- **Geometry + Tooling:** The engine extracts the profile (Z for length, X for radius). It checks the `Insert.corner_radius`. It knows the tool physically cannot cut a sharp 90° internal step.
- **Strategy:** It generates a `G71` roughing cycle. The `G71` block requires a `U` value (Depth of cut) and `F` value (Feed).
- **Postprocessor:** Lathe X-axis is programmed in **Diameter** (X = Radius * 2), whereas mills use standard Cartesian coordinates.
