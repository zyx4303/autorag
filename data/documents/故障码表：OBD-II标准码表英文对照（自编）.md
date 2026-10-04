# OBD-II Standard P-Code List (English · self-authored)

> **About this file**: OBD-II powertrain trouble codes and their standard descriptions are defined by
> public technical standards (SAE J2012 / ISO 15031-6); the code numbers and their meanings are
> **public technical facts**. The descriptions below are **written by the author of this project**
> for RAG demonstration purposes — no third-party code list has been copied.
> Always confirm diagnostics against the official service information for the specific vehicle.

OBD-II (On-Board Diagnostics II) is mandatory on light-duty vehicles sold in the United States
from model year 1996 onward. Codes follow a fixed structure so that any scan tool can read them.

---

## 1. Code structure

A DTC consists of **one letter + four digits**, for example **P0279**.

| Position | Meaning |
|---|---|
| 1st (letter) | **P** = Powertrain · **B** = Body · **C** = Chassis · **U** = Network communication |
| 2nd (digit) | **0** = generic (SAE-defined, same on all makes) · **1** = manufacturer-specific |
| 3rd (digit) | Subsystem, see table below |
| 4th–5th | Specific fault index |

### Third digit (subsystem)

| Value | Subsystem |
|---|---|
| 0 | Fuel, air or emission control |
| 1 | Fuel and air metering |
| 2 | Fuel injector circuit |
| 3 | Ignition system / misfire |
| 4 | Auxiliary emission control |
| 5 | Vehicle speed and idle control |
| 6 | Computer output circuit |
| 7 / 8 / 9 | Transmission |
| A / B / C | Hybrid propulsion |

### P-code ranges

| Range | System |
|---|---|
| P0001–P0099 | Fuel, air or emission control |
| P0100–P0199 | Fuel or air |
| P0200–P0299 | Fuel or air (injectors) |
| P0300–P0399 | Ignition or misfire |
| P0400–P0499 | Emission control |
| P0500–P0599 | Vehicle speed and idle control |
| P0600–P0699 | Computer or auxiliary output |
| P0700–P0799 | Transmission |
| P0800–P0899 | Transmission |
| P0900–P0999 | Transmission |
| P0A00–P0AFF | Hybrid propulsion |

---

## 2. Common abbreviations

| Term | Meaning |
|---|---|
| DTC | Diagnostic Trouble Code |
| MIL | Malfunction Indicator Lamp |
| ECM / PCM | Engine / Powertrain Control Module |
| TCM | Transmission Control Module |
| HO2S / O2S | Heated Oxygen Sensor / Oxygen Sensor |
| MAF / MAP | Mass Air Flow / Manifold Absolute Pressure |
| IAT / ECT | Intake Air Temperature / Engine Coolant Temperature |
| VVT | Variable Valve Timing |
| EVAP | Evaporative Emission Control System |
| EGR | Exhaust Gas Recirculation |
| Bank 1 / Bank 2 | Cylinder bank containing cylinder 1 / the other bank |
| Sensor 1 / Sensor 2 | First / second sensor in exhaust flow order |

---

## 3. Generic P-codes (SAE-defined, all manufacturers)

### P0001 – P0299 · Fuel, air or emission control

| Code | Description |
|---|---|
| P0001 | Fuel volume regulator control circuit open |
| P0002 | Fuel volume regulator control circuit range/performance |
| P0003 | Fuel volume regulator control circuit low |
| P0004 | Fuel volume regulator control circuit high |
| P0010 | "A" camshaft position actuator circuit |
| P0011 | "A" camshaft position timing over-advanced or system performance |
| P0012 | "A" camshaft position timing over-retarded |
| P0016 | Crankshaft / camshaft position correlation (Bank 1) |
| P0030 | HO2S heater control circuit (Bank 1, Sensor 1) |
| P0031 | HO2S heater control circuit low (Bank 1, Sensor 1) |
| P0032 | HO2S heater control circuit high (Bank 1, Sensor 1) |
| P0036 | HO2S heater control circuit (Bank 1, Sensor 2) |
| P0068 | MAP/MAF to throttle position correlation |
| P0069 | Manifold absolute pressure to barometric pressure correlation |
| P0070 | Ambient air temperature sensor circuit |
| P0087 | Fuel rail/system pressure too low |
| P0088 | Fuel rail/system pressure too high |
| P0100 | Mass or volume air flow circuit |
| P0101 | Mass or volume air flow circuit range/performance |
| P0102 | Mass or volume air flow circuit low input |
| P0103 | Mass or volume air flow circuit high input |
| P0105 | MAP/barometric pressure circuit |
| P0106 | MAP/barometric pressure circuit range/performance |
| P0107 | MAP/barometric pressure circuit low input |
| P0108 | MAP/barometric pressure circuit high input |
| P0110 | Intake air temperature sensor 1 circuit |
| P0115 | Engine coolant temperature circuit |
| P0116 | Engine coolant temperature circuit range/performance |
| P0117 | Engine coolant temperature circuit low |
| P0118 | Engine coolant temperature circuit high |
| P0120 | Throttle/pedal position sensor "A" circuit |
| P0121 | Throttle/pedal position sensor "A" circuit range/performance |
| P0122 | Throttle/pedal position sensor "A" circuit low |
| P0123 | Throttle/pedal position sensor "A" circuit high |
| P0125 | Insufficient coolant temperature for closed loop fuel control |
| P0128 | Coolant thermostat below regulating temperature |
| P0130 | O2 sensor circuit (Bank 1, Sensor 1) |
| P0131 | O2 sensor circuit low voltage (Bank 1, Sensor 1) |
| P0132 | O2 sensor circuit high voltage (Bank 1, Sensor 1) |
| P0133 | O2 sensor circuit slow response (Bank 1, Sensor 1) |
| P0134 | O2 sensor circuit no activity detected (Bank 1, Sensor 1) |
| P0135 | O2 sensor heater circuit (Bank 1, Sensor 1) |
| P0136 | O2 sensor circuit (Bank 1, Sensor 2) |
| P0137 | O2 sensor circuit low voltage (Bank 1, Sensor 2) |
| P0138 | O2 sensor circuit high voltage (Bank 1, Sensor 2) |
| P0139 | O2 sensor circuit slow response (Bank 1, Sensor 2) |
| P0140 | O2 sensor circuit no activity detected (Bank 1, Sensor 2) |
| P0141 | O2 sensor heater circuit (Bank 1, Sensor 2) |
| P0150 | O2 sensor circuit (Bank 2, Sensor 1) |
| P0155 | O2 sensor heater circuit (Bank 2, Sensor 1) |
| P0171 | System too lean (Bank 1) |
| P0172 | System too rich (Bank 1) |
| P0174 | System too lean (Bank 2) |
| P0175 | System too rich (Bank 2) |
| P0195 | Engine oil temperature sensor |
| P0200 | Injector circuit open |
| P0201 | Injector circuit open — cylinder 1 |
| P0202 | Injector circuit open — cylinder 2 |
| P0203 | Injector circuit open — cylinder 3 |
| P0204 | Injector circuit open — cylinder 4 |
| P0217 | Engine coolant over-temperature condition |
| P0219 | Engine overspeed condition |
| P0230 | Fuel pump primary circuit |
| P0234 | Turbo/supercharger overboost condition |
| P0261 | Cylinder 1 injector circuit low |
| P0262 | Cylinder 1 injector circuit high |
| P0263 | Cylinder 1 contribution/balance |
| P0264 | Cylinder 2 injector circuit low |
| P0265 | Cylinder 2 injector circuit high |
| P0266 | Cylinder 2 contribution/balance |
| P0299 | Turbo/supercharger underboost |

### P0300 – P0399 · Ignition or misfire

| Code | Description |
|---|---|
| P0300 | Random or multiple cylinder misfire detected |
| P0301 | Cylinder 1 misfire detected |
| P0302 | Cylinder 2 misfire detected |
| P0303 | Cylinder 3 misfire detected |
| P0304 | Cylinder 4 misfire detected |
| P0305 | Cylinder 5 misfire detected |
| P0306 | Cylinder 6 misfire detected |
| P0307 | Cylinder 7 misfire detected |
| P0308 | Cylinder 8 misfire detected |
| P0316 | Misfire detected on start-up |
| P0325 | Knock sensor 1 circuit (Bank 1) |
| P0326 | Knock sensor 1 circuit range/performance (Bank 1) |
| P0327 | Knock sensor 1 circuit low (Bank 1) |
| P0328 | Knock sensor 1 circuit high (Bank 1) |
| P0335 | Crankshaft position sensor "A" circuit |
| P0336 | Crankshaft position sensor "A" circuit range/performance |
| P0337 | Crankshaft position sensor "A" circuit low |
| P0338 | Crankshaft position sensor "A" circuit high |
| P0340 | Camshaft position sensor "A" circuit (Bank 1) |
| P0341 | Camshaft position sensor "A" circuit range/performance (Bank 1) |
| P0345 | Camshaft position sensor "A" circuit (Bank 2) |
| P0351 | Ignition coil "A" primary/secondary circuit |
| P0352 | Ignition coil "B" primary/secondary circuit |
| P0353 | Ignition coil "C" primary/secondary circuit |
| P0354 | Ignition coil "D" primary/secondary circuit |

### P0400 – P0499 · Emission control

| Code | Description |
|---|---|
| P0400 | Exhaust gas recirculation flow |
| P0401 | EGR flow insufficient |
| P0402 | EGR flow excessive |
| P0403 | EGR control circuit |
| P0404 | EGR control circuit range/performance |
| P0405 | EGR position sensor "A" circuit low |
| P0406 | EGR position sensor "A" circuit high |
| P0410 | Secondary air injection system |
| P0411 | Secondary air injection system incorrect flow |
| P0420 | Catalyst system efficiency below threshold (Bank 1) |
| P0430 | Catalyst system efficiency below threshold (Bank 2) |
| P0440 | Evaporative emission system |
| P0441 | Evaporative emission system incorrect purge flow |
| P0442 | Evaporative emission system small leak detected |
| P0443 | Evaporative emission purge control valve circuit |
| P0446 | Evaporative emission vent control circuit |
| P0455 | Evaporative emission system large leak detected |
| P0456 | Evaporative emission system very small leak detected |
| P0460 | Fuel level sensor circuit |
| P0463 | Fuel level sensor circuit high |
| P0480 | Cooling fan 1 control circuit |
| P0481 | Cooling fan 2 control circuit |

### P0500 – P0599 · Vehicle speed and idle control

| Code | Description |
|---|---|
| P0500 | Vehicle speed sensor "A" |
| P0501 | Vehicle speed sensor "A" range/performance |
| P0505 | Idle control system |
| P0506 | Idle control system RPM lower than expected |
| P0507 | Idle control system RPM higher than expected |
| P0510 | Closed throttle position switch |
| P0520 | Engine oil pressure sensor/switch circuit |
| P0521 | Engine oil pressure sensor range/performance |
| P0530 | A/C refrigerant pressure sensor circuit |
| P0562 | System voltage low |
| P0563 | System voltage high |
| P0571 | Brake switch "A" circuit |

### P0600 – P0699 · Computer or auxiliary output

| Code | Description |
|---|---|
| P0600 | Serial communication link |
| P0601 | Internal control module memory checksum error |
| P0602 | Control module programming error |
| P0603 | Internal control module keep-alive memory error |
| P0604 | Internal control module random access memory error |
| P0605 | Internal control module read only memory error |
| P0606 | ECM/PCM processor fault |
| P0607 | Control module performance |
| P0620 | Generator control circuit |
| P0621 | Generator lamp "L" terminal circuit |
| P0622 | Generator field "F" terminal circuit |
| P0645 | A/C clutch relay control circuit |
| P0650 | Malfunction indicator lamp control circuit |
| P0685 | ECM/PCM power relay control circuit |

### P0700 – P0999 · Transmission

| Code | Description |
|---|---|
| P0700 | Transmission control system (MIL request) |
| P0701 | Transmission control system range/performance |
| P0702 | Transmission control system electrical |
| P0703 | Brake switch "B" circuit |
| P0705 | Transmission range sensor circuit |
| P0706 | Transmission range sensor circuit range/performance |
| P0710 | Transmission fluid temperature sensor circuit |
| P0711 | Transmission fluid temperature sensor range/performance |
| P0712 | Transmission fluid temperature sensor circuit low |
| P0713 | Transmission fluid temperature sensor circuit high |
| P0715 | Input/turbine speed sensor "A" circuit |
| P0716 | Input/turbine speed sensor "A" circuit range/performance |
| P0717 | Input/turbine speed sensor "A" circuit no signal |
| P0720 | Output speed sensor circuit |
| P0721 | Output speed sensor circuit range/performance |
| P0722 | Output speed sensor circuit no signal |
| P0725 | Engine speed input circuit |
| P0730 | Incorrect gear ratio |
| P0731 | Gear 1 incorrect ratio |
| P0732 | Gear 2 incorrect ratio |
| P0733 | Gear 3 incorrect ratio |
| P0734 | Gear 4 incorrect ratio |
| P0740 | Torque converter clutch circuit |
| P0741 | Torque converter clutch stuck off or slipping |
| P0742 | Torque converter clutch stuck on |
| P0743 | Torque converter clutch circuit electrical |
| P0750 | Shift solenoid "A" |
| P0751 | Shift solenoid "A" performance or stuck off |
| P0752 | Shift solenoid "A" stuck on |
| P0753 | Shift solenoid "A" electrical |
| P0755 | Shift solenoid "B" |
| P0760 | Shift solenoid "C" |
| P0765 | Shift solenoid "D" |
| P0770 | Shift solenoid "E" |
| P0801 | Reverse inhibit control circuit |
| P0810 | Clutch position control |
| P0850 | Park/neutral switch input circuit |
| P0900 | Clutch actuator circuit |

### P0A00 – P0AFF · Hybrid propulsion (excerpt)

| Code | Description |
|---|---|
| P0A0F | Engine failed to start |
| P0A1F | Battery energy control module |
| P0A7F | Hybrid battery pack deterioration |
| P0A80 | Replace hybrid battery pack |
| P0A93 | Inverter cooling system performance |
| P0AFA | Hybrid battery system voltage low |

---

## 4. Manufacturer-specific codes (P1xxx)

Codes in the `P1xxx` range are **defined by each manufacturer independently** — the same number
can mean completely different things on different brands. Always consult the official service
information for the vehicle in question. Some manufacturers also use `P3000–P3399`.

| Code | Example meaning (illustrative only) |
|---|---|
| P1101 | Mass air flow / throttle correlation out of range (one make's definition) |
| P1102 | Mass air flow sensor signal below lower limit (one make's definition) |
| P1128 | Closed loop fuel correction above upper limit (one make's definition) |
| P1130 | Front O2 sensor slow response (one make's definition) |
| P1188 | Long term fuel trim too rich (one make's definition) |
| P1300 | Ignition misfire detection circuit fault (one make's definition) |
| P1336 | Crankshaft position learn value not stored (one make's definition) |
| P1404 | EGR valve position feedback out of range (one make's definition) |
| P1604 | Engine start performance fault (one make's definition) |
| P1682 | Ignition switch supply circuit fault (one make's definition) |
