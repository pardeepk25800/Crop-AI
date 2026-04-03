import os
import torch
import numpy as np
from PIL import Image
from disease_model import DiseasePredictor

def test_predictor():
    print("Testing DiseasePredictor dynamic response...")
    try:
        predictor = DiseasePredictor()
        
        # Test 1: Solid Green Image (likely to be 'healthy' or something specific)
        img_green = Image.new("RGB", (224, 224), color=(34, 139, 34))
        res1 = predictor.predict(img_green)
        print(f"Test 1 (Green): {res1['predicted_class']} ({res1['confidence']}%)")
        
        # Test 2: Solid Red Image (should yield different results)
        img_red = Image.new("RGB", (224, 224), color=(255, 0, 0))
        res2 = predictor.predict(img_red)
        print(f"Test 2 (Red): {res2['predicted_class']} ({res2['confidence']}%)")
        
        # Test 3: Random Noise
        img_noise = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        res3 = predictor.predict(img_noise)
        print(f"Test 3 (Noise): {res3['predicted_class']} ({res3['confidence']}%)")
        
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_predictor()
