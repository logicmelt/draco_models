This package reads data from an Influx database and parses it for training. Raw data should have at least the following keys:

+ EventID: Integer that associates a muon to a source proton (From the simulation).
+ process_ID: Integer indicating which process generated the muon.
+ run_id: UUID for different runs.
+ phi: Azimuthal angle, takes values between  and 2pi.
+ theta: Zenithal angle, takes values between  and pi/2.
+ density_day_idx: Index that can be used to identify the used density profile.