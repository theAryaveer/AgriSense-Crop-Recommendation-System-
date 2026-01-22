import requests
import json

# Test the Advanced Analysis feature
url = "http://127.0.0.1:5000/predict"

# Sample soil data
test_data = {
    "N": 90,
    "P": 45,
    "K": 50,
    "temperature": 28,
    "humidity": 75,
    "ph": 6.8,
    "rainfall": 120,
    "language": "en"
}

print("Testing Advanced Analysis Feature...")
print("=" * 60)
print(f"\nSending request to: {url}")
print(f"Input soil data: {json.dumps(test_data, indent=2)}")
print("\n" + "=" * 60)

try:
    response = requests.post(url, json=test_data)
    result = response.json()
    
    if result.get('success'):
        print("\n✅ SUCCESS! Advanced Analysis Working!\n")
        
        # Display top prediction
        top = result['top_prediction']
        print(f"🏆 TOP RECOMMENDATION: {top['crop']} ({top['confidence']}% confidence)")
        print(f"\n   Nutrient Gaps:")
        print(f"   - Nitrogen (N): {top['analysis']['gap']['N']} kg/ha")
        print(f"   - Phosphorus (P): {top['analysis']['gap']['P']} kg/ha")
        print(f"   - Potassium (K): {top['analysis']['gap']['K']} kg/ha")
        print(f"\n   Fertilizer Needed (50kg bags):")
        print(f"   - Urea: {top['analysis']['bags']['urea']} bags")
        print(f"   - SSP: {top['analysis']['bags']['ssp']} bags")
        print(f"   - MOP: {top['analysis']['bags']['mop']} bags")
        print(f"\n   Economics:")
        print(f"   - Expected Yield: {top['analysis']['economics']['yield']} kg/ha")
        print(f"   - Market Price: ₹{top['analysis']['economics']['price']}/kg")
        print(f"   - Total Revenue: ₹{top['analysis']['economics']['revenue']:,}")
        
        # Display alternatives
        if result.get('alternatives'):
            print(f"\n📋 ALTERNATIVES:")
            for i, alt in enumerate(result['alternatives'], 1):
                print(f"\n   {i}. {alt['crop']} ({alt['confidence']}% confidence)")
                print(f"      Revenue: ₹{alt['analysis']['economics']['revenue']:,}")
                print(f"      Fertilizer: {alt['analysis']['bags']['urea']} Urea, "
                      f"{alt['analysis']['bags']['ssp']} SSP, "
                      f"{alt['analysis']['bags']['mop']} MOP")
        
        print("\n" + "=" * 60)
        print("Full Response (JSON):")
        print(json.dumps(result, indent=2))
        
    else:
        print(f"\n❌ Error: {result.get('error')}")
        
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("Make sure Flask server is running on http://127.0.0.1:5000")
