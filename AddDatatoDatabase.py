
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("tollsdb.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://tollsdb-a0b14-default-rtdb.firebaseio.com/"
})


ref = db.reference('Drivers')

data = {
    "88737":
        {
            "name": "Elan Mask",
            "Car_Type": "Townace",
            "Route": "KTS - CNM",
            "starting_year": 2023,
            "total_attendance": 4,
            "Tolls_amount": 50,
            "last_attendance_time": "2023-10-11 00:54:34"
        },
    "89386":
        {
            "name": "Hadi Azizi",
            "Car_Type": "Townace",
            "Route": "KTS - CNM",
            "starting_year": 2023,
            "total_attendance": 5,
            "Tolls_amount": 50,
            "last_attendance_time": "2023-10-15 00:54:34"
        },
    "94341":
        {
            "name": "Mohmood Ahmadi",
            "Car_Type": "Mercidise",
            "Route": "KTS - CNM",
            "starting_year": 2023,
            "total_attendance": 2,
            "Tolls_amount": 50,
            "last_attendance_time": "2023-10-12 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)
