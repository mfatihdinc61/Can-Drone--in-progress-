# import cv2
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration
#
# # Load the BLIP model and processor for image captioning
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#
# # Replace with your DroidCam IP address (e.g., http://192.168.1.10:4747/video)
# droidcam_url = "http://192.168.1.204:4747/video"
#
# # Initialize the video capture from DroidCam
# cap = cv2.VideoCapture(droidcam_url)
#
# if not cap.isOpened():
#     print("Error: Could not open DroidCam stream.")
#     exit()
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     # Convert the frame to RGB for BLIP processing
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the image with BLIP to generate a description
#     inputs = blip_processor(images=frame_rgb, return_tensors="pt")
#     with torch.no_grad():
#         generated_ids = blip_model.generate(**inputs)
#         generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#
#     # Display the description on the frame
#     cv2.putText(frame, f"Description: {generated_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#
#     # Display the resulting frame with the description
#     cv2.imshow('Scene Description (DroidCam)', frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()










"""
version 3
"""

import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Load the BLIP model and processor for image captioning
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Replace with your DroidCam IP address (e.g., http://192.168.x.xxx:4747/video)
droidcam_url = "http://192.168.1.203:4747/video"

# Initialize the video capture from DroidCam
cap = cv2.VideoCapture(droidcam_url)

if not cap.isOpened():
    print("Error: Could not open DroidCam stream.")
    exit()

# Store the time of the last description generation
last_description_time = 0
description_interval = 3  # seconds

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get the current time
    current_time = time.time()

    # Generate scene description only if 3 seconds have passed since the last one
    if current_time - last_description_time >= description_interval:
        # Convert the frame to RGB for BLIP processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with BLIP to generate a description
        inputs = blip_processor(images=frame_rgb, return_tensors="pt")
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs)
            generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Print the scene description in the console
        print(f"Scene Description: {generated_text}")

        # Update the time of the last description
        last_description_time = current_time

    # Display the frame without any text
    cv2.imshow('DroidCam Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()









"""
version 4
"""

# import cv2
# import time
# import torch
# import openai
# from djitellopy import Tello
# from transformers import BlipProcessor, BlipForConditionalGeneration
#
# # Initialize OpenAI API
# openai.api_key = 'your_openai_api_key'
#
# # Initialize the BLIP model and processor for image captioning
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#
# # Initialize Tello drone
# tello = Tello()
# tello.connect()
# tello.streamon()
#
# # Time tracker for sending scene descriptions
# last_description_time = 0
# description_interval = 3  # Seconds
# last_command = None  # Store the last command given to the drone
#
#
# # Function to send scene description and get command from ChatGPT
# def send_to_chatgpt(scene_description):
#     prompt = f"Scene: {scene_description}. Based on this, what should the drone do next?"
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=150
#     )
#     return response.choices[0].text.strip()
#
#
# # Main loop
# while True:
#     # Capture frame from the drone
#     frame_read = tello.get_frame_read()
#     frame = frame_read.frame
#
#     # Get the current time
#     current_time = time.time()
#
#     # Generate scene description every 3 seconds
#     if current_time - last_description_time >= description_interval:
#         # Convert the frame to RGB for BLIP processing
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Generate scene description
#         inputs = blip_processor(images=frame_rgb, return_tensors="pt")
#         with torch.no_grad():
#             generated_ids = blip_model.generate(**inputs)
#             scene_description = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#
#         print(f"Scene Description: {scene_description}")
#
#         # Send scene description to ChatGPT
#         action_command = send_to_chatgpt(scene_description)
#
#         # Store the action command for future use
#         last_command = action_command
#
#         # Print the command for the drone to perform
#         print(f"Command: {action_command}")
#
#         # Execute the command (simple example)
#         if 'move up' in action_command.lower():
#             tello.move_up(20)
#         elif 'move down' in action_command.lower():
#             tello.move_down(20)
#         elif 'rotate left' in action_command.lower():
#             tello.rotate_counter_clockwise(45)
#         elif 'rotate right' in action_command.lower():
#             tello.rotate_clockwise(45)
#
#         # Update last description time
#         last_description_time = current_time
#
#     # Check if a key (command) is pressed, if no key is pressed, continue with last command
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         tello.land()
#         break
#     elif key != -1:
#         user_input = input("Enter a command for the drone (or press Enter to skip):")
#         if user_input:
#             # Send user command to ChatGPT for processing
#             action_command = send_to_chatgpt(user_input)
#             last_command = action_command
#
#             # Execute the new command
#             if 'move up' in action_command.lower():
#                 tello.move_up(20)
#             elif 'move down' in action_command.lower():
#                 tello.move_down(20)
#             elif 'rotate left' in action_command.lower():
#                 tello.rotate_counter_clockwise(45)
#             elif 'rotate right' in action_command.lower():
#                 tello.rotate_clockwise(45)
#
# # Release resources
# tello.streamoff()
# cv2.destroyAllWindows()


"""
v-5
"""
#
# import cv2
# import time
# import torch
# import openai
# import threading
# import re
# from djitellopy import Tello
# from transformers import BlipProcessor, BlipForConditionalGeneration
#
# flag = 0
#
#
# import logging
# # Set logging level to WARNING to hide INFO messages
# logging.getLogger("djitellopy").setLevel(logging.WARNING)
#
#
# # Initialize OpenAI API
# # openai.api_key = 'your_openai_api_key'
# openai.api_key = "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"
#
#
# # Initialize the BLIP model and processor for image captioning
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#
# # Initialize Tello drone
# tello = Tello()
# tello.connect()
# tello.streamon()
#
# # Time tracker for sending scene descriptions
# last_description_time = 0
# description_interval = 3  # Seconds
# last_command = None  # Store the last command given to the drone
#
# # Variable to store user input
# user_input = None
# input_thread_running = True
#
# # Function to handle user input in a separate thread
# def get_user_input():
#     global user_input, input_thread_running
#     while input_thread_running:
#         user_input = input("Enter a command for the drone (or press Enter to skip):").strip()
#         if user_input.lower() == 'q':
#             input_thread_running = False
#             break
#
# # Start user input thread
# input_thread = threading.Thread(target=get_user_input)
# input_thread.start()
#
# # Function to send scene description and get command from ChatGPT
# def send_to_chatgpt(scene_description):
#     # prompt = f"Scene: {scene_description}. Based on this, what should the drone do next? Please provide the command in Python code format enclosed in ```python ... ```."
#     prompt = f"Scene: {scene_description}. if the last certain command requires you to take an action for drone, return the code snippet in the following format ```python ... ```."
#
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=100
#     )
#     return response.choices[0].message['content'].strip()
#
# # Function to extract code from ChatGPT response
# def extract_code(response):
#     match = re.search(r'```python(.*?)```', response, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return None
#
# # Main loop
# while input_thread_running:
#     # Capture frame from the drone
#     frame_read = tello.get_frame_read()
#     frame = frame_read.frame
#
#     # Get the current time
#     current_time = time.time()
#
#     # Generate scene description every 3 seconds
#     if current_time - last_description_time >= description_interval:
#         # Convert the frame to RGB for BLIP processing
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Generate scene description
#         inputs = blip_processor(images=frame_rgb, return_tensors="pt")
#         with torch.no_grad():
#             generated_ids = blip_model.generate(**inputs)
#             scene_description = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#
#         print(f"Scene Description: {scene_description}")
#
#         # Send scene description to ChatGPT
#         action_command = send_to_chatgpt(scene_description)
#
#         # Extract code using regex
#         code_to_execute = extract_code(action_command)
#         if code_to_execute:
#             try:
#                 exec(code_to_execute)
#             except Exception as e:
#                 print(f"Error executing command from scene description: {e}")
#
#         # Update last description time
#         last_description_time = current_time
#
#     # Check if there's new user input
#
#     if user_input:
#         # Read content from theprepromt.txt
#         with open('theprepromt.txt', 'r') as file:
#             pre_prompt_content = file.read().strip()
#
#         # Combine the pre-prompt with user input
#         combined_prompt = f"{pre_prompt_content}\n{user_input}"
#
#         # Send combined command to ChatGPT for processing
#         action_command = send_to_chatgpt(combined_prompt)
#
#         # Extract code using regex
#         code_to_execute = extract_code(action_command)
#         if code_to_execute:
#             try:
#                 exec(code_to_execute)
#             except Exception as e:
#                 print(f"Error executing user command: {e}")
#
#         # Clear user input after processing
#         user_input = None
#
#         flag = 1
#
#     # Make the drone hover
#     tello.send_rc_control(0, 0, 0, 0)
#     time.sleep(1)
#
# # Clean up resources
# tello.streamoff()
# cv2.destroyAllWindows()
# input_thread_running = False
# input_thread.join()