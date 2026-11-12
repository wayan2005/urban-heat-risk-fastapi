from app import predict_heat_risk

hot_area  = (7, 90, 5, 0.10, 20000, 30, 0.10, 38, 50, 1.0)
mild_area = (7, 20, 55, 0.50, 1500, 6, 0.22, 24, 60, 5.0)

for name, area in [("Hot dense area", hot_area), ("Green mild area", mild_area)]:
    result, probs = predict_heat_risk(*area)
    print(name, "->", result, {k: round(float(v), 2) for k, v in probs.items()})
