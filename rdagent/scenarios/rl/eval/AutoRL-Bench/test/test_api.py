"""
Test API endpoints

Usage:
    # First start server: python -m autorl_bench.server
    # Then run: python test/test_api.py
"""

import requests

BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint"""
    print("Testing GET /")
    resp = requests.get(f"{BASE_URL}/")
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    return resp.status_code == 200


def test_scenarios():
    """Test list scenarios"""
    print("\nTesting GET /scenarios")
    resp = requests.get(f"{BASE_URL}/scenarios")
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Scenarios: {[s['id'] for s in data['scenarios']]}")
    return resp.status_code == 200


def test_scenario_detail():
    """Test get scenario detail"""
    print("\nTesting GET /scenarios/gsm8k")
    resp = requests.get(f"{BASE_URL}/scenarios/gsm8k")
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Name: {data.get('name')}")
    print(f"  Model: {data.get('base_model')}")
    print(f"  Train data: {data.get('train_data')}")
    return resp.status_code == 200


def main():
    print("=" * 50)
    print("API Test")
    print("=" * 50)
    
    results = []
    results.append(("Root", test_root()))
    results.append(("Scenarios", test_scenarios()))
    results.append(("Scenario Detail", test_scenario_detail()))
    
    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    print("=" * 50)


if __name__ == "__main__":
    main()

