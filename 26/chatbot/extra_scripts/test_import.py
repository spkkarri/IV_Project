import sys
import os
# Add chatbot directory to path
sys.path.append(r"c:\Users\mehul\OneDrive\Desktop\Major_Project\chatbot")

try:
    import matlab.engine
    print("matlab.engine is installed")
except ImportError:
    print("matlab.engine is NOT installed")

agents_dir = r"c:\Users\mehul\OneDrive\Desktop\Major_Project\chatbot\agents"
sys.path.append(agents_dir)
try:
    import matlab_executor_agent
    print("Successfully imported matlab_executor_agent directly")
    run_matlab_executor_agent = matlab_executor_agent.run_matlab_executor_agent
    print("Running a test question...")
    result = run_matlab_executor_agent("Plot the step response of G(s) = 1/(s+2)")
    print("Agent Response length:", len(result))
    if "data:image/png;base64," in result:
        print("SUCCESS: Plot found in response")
    else:
        print("FAILURE: No plot found in response")
        print("Response start:", result[:500])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
