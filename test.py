# hello_world.py
print("Hello World from FuzzBall!")
print("Script is running successfully!")

# Test that arguments work
import sys
if len(sys.argv) > 1:
    print(f"Received arguments: {sys.argv[1:]}")