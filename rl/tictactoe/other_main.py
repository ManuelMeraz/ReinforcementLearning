#! /usr/bin/env python3
import signal


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)


if __name__ == "__main__":
    main()
