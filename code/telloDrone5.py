import openai
from djitellopy import Tello
import cv2
import threading
import re

# Initialize the Tello object
tello = Tello()

# Set up OpenAI API
openai.api_key = ":)"


with open('prepromt.txt', 'r') as file:
    file_content = file.read()


def connect_tello():
    try:
        tello.connect()
        print(f"Battery Life: {tello.get_battery()}%")
    except:
        print("Bağlantı başarısız")


# def stream_video():
#     tello.streamon()
#     frame_read = tello.get_frame_read()
#
#     while True:
#         frame = frame_read.frame
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         cv2.imshow("Tello Stream", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     tello.streamoff()
#     cv2.destroyAllWindows()


def stream_video():

    try:
        tello.streamon()
        frame_read = tello.get_frame_read()

    except:
        print("Stream-on failed")

    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the current width and height of the frame
        height, width = frame.shape[:2]

        # Divide the width and height by two
        new_width = width // 2
        new_height = height // 2

        # Resize the frame to the new dimensions
        frame = cv2.resize(frame, (new_width, new_height))

        cv2.imshow("Tello Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()



# def get_chatgpt_response(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=100
#     )
#     return response.choices[0].text.strip()



def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()


# def execute_command(command):
#     if "takeoff" in command:
#         tello.takeoff()
#     elif "land" in command:
#         tello.land()
#     elif "forward" in command:
#         distance = int(command.split(" ")[1])
#         tello.move_forward(distance)
#     elif "backward" in command:
#         distance = int(command.split(" ")[1])
#         tello.move_back(distance)
#     elif "up" in command:
#         distance = int(command.split(" ")[1])
#         tello.move_up(distance)
#     elif "down" in command:
#         distance = int(command.split(" ")[1])
#         tello.move_down(distance)
#     elif "right" in command:
#         distance = int(command.split(" ")[1])
#         tello.move_right(distance)
#     elif "left" in command:
#         distance = int(command.split(" ")[1])
#         tello.move_left(distance)
#     elif "rotate" in command:
#         angle = int(command.split(" ")[1])
#         tello.rotate_clockwise(angle)
#     elif "battery" in command:
#         print(f"Battery Life: {tello.get_battery()}%")
#         if tello.get_battery() < 20:
#             tello.land()
#     else:
#         print("Command not recognized!")


def direct_execute_command(command):
    # Regular expression to extract code after ```python until the end of the string or next ```
    pattern = r"```python\s*(.*)```"
    match = re.search(pattern, command, re.DOTALL)

    if match:
        code_string = match.group(1).strip()
        print(code_string)
        try:
            exec(code_string)
        except:
            print("There is a problem in code's format, re-write")
    else:
        print("No command was taken from ChatGPT")


def command_tello():
    while True:
        prompt = input("Enter a command for ChatGPT: ")
        prompt =  file_content + prompt
        if prompt.lower() == "exit":
            break

        chatgpt_response = get_chatgpt_response(prompt)
        print(f"ChatGPT Response: {chatgpt_response}")

        direct_execute_command(chatgpt_response)


def main():
    # Connect to Tello
    connect_tello()

    # Start video streaming in a separate thread
    video_thread = threading.Thread(target=stream_video)
    video_thread.start()

    # Start command input loop
    command_tello()

    # Wait for the video thread to finish
    video_thread.join()

    # Land the drone if it's still flying
#    tello.land()


if __name__ == "__main__":
    main()