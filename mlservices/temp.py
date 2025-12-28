import joblib

crop_encoder = joblib.load("models/crop_label_encoder.pkl")

print("Allowed crops:")
print(crop_encoder.classes_)
