"""
Test script for the Ybus calculation agent
"""
from agents.power_flow_agent import run_power_flow_agent

# Test 1: Simple Ybus calculation with the example from MATLAB code
test_query_1 = """
Calculate the Ybus matrix for a 3-bus system with the following branch data:

Branch 1: From bus 1 to bus 2
- Resistance: 0.03 pu
- Reactance: 0.08 pu
- Transformer ratio: 1
- Shunt admittance: 0.04 pu

Branch 2: From bus 1 to bus 3
- Resistance: 0.02 pu
- Reactance: 0.05 pu
- Transformer ratio: 1
- Shunt admittance: 0.02 pu

Branch 3: From bus 2 to bus 3
- Resistance: 0.01 pu
- Reactance: 0.03 pu
- Transformer ratio: 1
- Shunt admittance: 0.03 pu

Please calculate and display the Ybus matrix.
"""

# Test 2: Ybus calculation followed by power flow
test_query_2 = """
I have a 3-bus power system with the following branch data:
- Branch 1-2: R=0.03, X=0.08, a=1, shunt=0.04
- Branch 1-3: R=0.02, X=0.05, a=1, shunt=0.02
- Branch 2-3: R=0.01, X=0.03, a=1, shunt=0.03

First calculate the Ybus matrix, then solve power flow if:
- Bus 1 is slack bus with V=1.0∠0°
- Bus 2 has P=0.5 pu load
- Bus 3 has P=0.3 pu load
"""

if __name__ == "__main__":
    print("="*80)
    print("Test 1: Simple Ybus Calculation")
    print("="*80)
    result1 = run_power_flow_agent(test_query_1)
    print(result1)
    
    print("\n\n")
    print("="*80)
    print("Test 2: Ybus + Power Flow Analysis")
    print("="*80)
    result2 = run_power_flow_agent(test_query_2)
    print(result2)

