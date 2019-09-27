#!/usr/bin/env python3

class Human(object):
    def __init__(self, mark):
        self.mark = mark

    def next_action(self, available_actions):
        while True:
            user_input = input("Enter move[1-9]: ")

            try:
                action = int(user_input) - 1

                if action not in available_actions:
                    raise ValueError()

            except ValueError:
                print("Illegal location: '{}'".format(user_input))
            else:
                break

        return action
