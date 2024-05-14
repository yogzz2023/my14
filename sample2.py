import numpy as np
import matplotlib.pyplot as plt
import csv
import math

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def Filter_state_covariance(self, measurements, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

        # Update step with JPDA
        Z = np.array(measurements)
        H = np.eye(3, 6)
        Inn = Z - np.dot(H, self.Sf)  # Calculate innovation directly
        S = np.dot(H, np.dot(self.pf, H.T)) + self.R
        K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn.T)
        self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)

        # Calculate association probabilities using JPDA
        association_probs = self.calculate_association_probabilities(Z)

        # Find the most likely associated measurement for the track with the highest marginal probability
        max_associated_index = np.argmax(association_probs)
        most_likely_associated_measurement = measurements[max_associated_index]

        # Convert filtered state to spherical coordinates
        r_pred, az_pred, el_pred = cart2sph(self.Sf[0][0], self.Sf[1][0], self.Sf[2][0])
        
        # Print predicted range, azimuth, and elevation
        print("Predicted Range:", r_pred)
        print("Predicted Azimuth:", az_pred)
        print("Predicted Elevation:", el_pred)

        print("Filtered Time:", current_time)

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf, most_likely_associated_measurement

    def calculate_association_probabilities(self, measurements):
        num_measurements = len(measurements)
        num_tracks = 1  # For simplicity, assuming only one track
        conditional_probs = np.zeros((num_measurements, num_tracks))

        # Calculate conditional probabilities using JPDA
        for i in range(num_measurements):
            for j in range(num_tracks):
                # Here you can implement your conditional probability calculation using JPDA
                # For simplicity, let's assume equal conditional probabilities for now
                conditional_probs[i, j] = 1.0 / num_measurements

        # Calculate marginal probabilities
        marginal_probs = np.sum(conditional_probs, axis=1) / num_tracks

        # Calculate joint probabilities
        joint_probs = conditional_probs * marginal_probs[:, np.newaxis]

        # Calculate association probabilities
        association_probs = np.sum(joint_probs, axis=0)

        return association_probs

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):

    #x = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
    #y = r * np.cos(np.radians(el)) * np.cos(np.radians(az))
    #z = r * np.sin(np.radians(el))
    
    x = r * np.cos(el * 3.14/180) * np.sin(az * 3.14/180)
    y = r * np.cos(el * 3.14/180) * np.cos(az * 3.14/180)
    z = r * np.sin(el * 3.14/180)
    
    print("filtered x: ",x)
    print("filtered y: ",y)
    print("filtered z: ",z)
    

    r = math.sqrt(x**2 + y**2 + z**2)
    el = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    az = math.degrees(math.atan2(y, x))
    
    
    if x > 0.0:
        az = 90 - az
    else:
        az = 270 - az
        
    print("Predicted Range:", r)
    print("Predicted Azimuth:", el)
    print("Predicted Elevation:", az)
    
    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_57.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Step 1: Initialize with 1st measurement M1
kalman_filter.Initialize_Filter_state_covariance(measurements[0][0], measurements[0][1], measurements[0][2], 0, 0, 0, measurements[0][3])

# Step 2: Initialize with 2nd measurement M2
kalman_filter.Initialize_Filter_state_covariance(measurements[1][0], measurements[1][1], measurements[1][2], 0, 0, 0, measurements[1][3])

# Lists to store predicted values
filtered_x = []
filtered_y = []
filtered_z = []
filtered_time = []

# Process measurements and get predicted state estimates at each time step
for i in range(2, len(measurements)):
    # Step 3: Get the velocity from step 1 and 2
    vel_x = (measurements[1][0] - measurements[0][0]) / (measurements[1][3] - measurements[0][3])
    vel_y = (measurements[1][1] - measurements[0][1]) / (measurements[1][3] - measurements[0][3])
    vel_z = (measurements[1][2] - measurements[0][2]) / (measurements[1][3] - measurements[0][3])

    # Step 4: Get measurement M3
    predicted_time = measurements[i][3]
    predicted_measurements = (measurements[i][0] + vel_x * (predicted_time - measurements[1][3]), 
                               measurements[i][1] + vel_y * (predicted_time - measurements[1][3]), 
                               measurements[i][2] + vel_z * (predicted_time - measurements[1][3]))
    
    # Step 5: Do association using JPDA
    filtered_state, most_likely_associated_measurement = kalman_filter.Filter_state_covariance([predicted_measurements], predicted_time)

    # Append filtered values to lists
    filtered_x.append(filtered_state[0][0])
    filtered_y.append(filtered_state[1][0])
    filtered_z.append(filtered_state[2][0])
    filtered_time.append(predicted_time)

# Plotting filtered x, y, and z components against time
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(filtered_time, filtered_x, label='Filtered X')
plt.xlabel('Time')
plt.ylabel('Filtered X')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(filtered_time, filtered_y, label='Filtered Y')
plt.xlabel('Time')
plt.ylabel('Filtered Y')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(filtered_time, filtered_z, label='Filtered Z')
plt.xlabel('Time')
plt.ylabel('Filtered Z')
plt.legend()

plt.tight_layout()
plt.show()
