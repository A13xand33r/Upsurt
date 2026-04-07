# apslpals
from mqtt_comm import publish_control

# LED brightness
led1_brightness = 255
led2_brightness = 255
led3_brightness = 255

def send_command(device, brightness=None, action=None):
    data = {"device": device}
    if brightness is not None:
        data["action"] = "on"
        data["speed"] = brightness
    elif action is not None:
        data["action"] = action
    else:
        return

    ok = publish_control(data)
    if ok:
        print(f"Published to {device}: {data}")
    else:
        print(f"MQTT publish failed for {device}: {data}")

while True:
    print("\nSelect an option:")
    print("1: Turn ON LED1")
    print("2: LED1 brightness down")
    print("3: LED1 brightness up")
    print("4: Turn OFF LED1")
    print("5: Turn ON LED2")
    print("6: LED2 brightness down")
    print("7: LED2 brightness up")
    print("8: Turn OFF LED2")
    print("9: Turn ON Motor")
    print("10: Turn OFF Motor")
    print("11: Turn ON LED3")
    print("12: LED3 brightness down")
    print("13: LED3 brightness up")
    print("14: Turn OFF LED3")
    print("0: Exit")

    choice = input("Enter option number: ")

    if choice == "0":
        print("Exiting...")
        break
    elif choice == "1":
        led1_brightness = 255
        send_command("led1", brightness=led1_brightness)
    elif choice == "2":
        led1_brightness = max(0, led1_brightness - 50)
        send_command("led1", brightness=led1_brightness)
    elif choice == "3":
        led1_brightness = max(0, led1_brightness + 50)
        send_command("led1", brightness=led1_brightness)
    elif choice == "4":
        led1_brightness = 0
        send_command("led1", action="off")
    elif choice == "5":
        led2_brightness = 255
        send_command("led2", brightness=led2_brightness)
    elif choice == "6":
        led2_brightness = max(0, led2_brightness - 50)
        send_command("led2", brightness=led2_brightness)
    elif choice == "7":
        led2_brightness = max(0, led2_brightness + 50)
        send_command("led2", brightness=led2_brightness)
    elif choice == "8":
        led2_brightness = 0
        send_command("led2", action="off")
    elif choice == "9":
        send_command("motor", action="off")
    elif choice == "10":
        send_command("motor", action="on")
    elif choice == "11":
        led1_brightness = 255
        send_command("led3", brightness=led1_brightness)
    elif choice == "12":
        led1_brightness = max(0, led1_brightness - 50)
        send_command("led3", brightness=led1_brightness)
    elif choice == "13":
        led1_brightness = max(0, led1_brightness + 50)
        send_command("led3", brightness=led1_brightness)
    elif choice == "14":
        led1_brightness = 0
        send_command("led3", action="off")
    else:
        print("Invalid option, try again.")
