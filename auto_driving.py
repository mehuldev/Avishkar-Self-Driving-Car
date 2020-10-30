"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import division, print_function, absolute_import
import argparse
import logging
import random
import time
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from PIL import Image

#tf.debugging.set_log_device_placement(True)
model = load_model("Models/model1.h5")
try:
    import pygame

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

i = 0
WINDOW_WIDTH = 320
WINDOW_HEIGHT = 240
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=15,
        NumberOfPedestrians=30,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    # camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    # camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera1.set_position(2.0, 0.0, 1.4)
    # camera1.set_rotation(0.0, 0.0, 0.0)
    # settings.add_sensor(camera1)
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class PID:
    def __init__(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.total = 0
        self.prev = 0


#pid = PID()

steer_pid = PID(kp=0.04,ki=0.1,kd=0.1)
throttle_pid = PID(kp=0.04,ki=0.1,kd=0.1)


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._enable_autopilot = args.autopilot
        self._map_view = None
        self._is_on_reverse = False
        self._display_map = args.map
        self._city_name = None
        self._map = None
        self._map_shape = None
        self._map_view = None
        self._position = None
        self._agent_positions = None

        ################################################################ EDIT BY MY

        ## - i is for data image sequence
        self._i = 0
        ## - val1 is for steering angle
        self._val1 = 0
        ## - val2 is for throttle
        self._val2 = 0
        ## - val3 is for data recording start
        self._val3 = 0
        ## - velocity of car
        self._velocity = 0
        ## - data
        self._data = {}

        ################################################################@

    def resetVal(self):
        self._val1 = 0
        self._val2 = 0
        self._is_on_reverse = False

    ##execute function is main loop
    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                self._on_loop()

        finally:
            ################################################################### SAVING THE DATA ON MEMORY
            os.chdir(
                'C:\\Users\\User\\Desktop\\CarlaSimulator\\PythonClient\\img')  ## SET FOLDER FOR STORING IMAGE DATA
            io.savemat('data', self._data)  # SAVING....
            print("Data Saved")
            pygame.quit()

            ####################################

    def _initialize_game(self):
        self._on_new_episode()

        if self._city_name is not None:
            self._map = CarlaMap(self._city_name, 0.1643, 50.0)
            self._map_shape = self._map.map_image.shape
            self._map_view = self._map.get_map(WINDOW_HEIGHT)

            extra_width = int((WINDOW_HEIGHT / float(self._map_shape[0])) * self._map_shape[1])
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + extra_width, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        if self._display_map:
            self._city_name = scene.map_name
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):

        measurements, sensor_data = self.client.read_data()
        control = VehicleControl()
        self._main_image = sensor_data.get('CameraRGB', None)
        # self._mini_view_image1 = sensor_data.get('CameraDepth', None)
        try:
            print("model starting")
            img = image_converter.to_rgb_array(self._main_image)
            x = [img]
            x = np.asarray(x)
            s, t = model.predict(x)[0]
            throttle_pid.total += t
            control.throttle = throttle_pid.kp * t + throttle_pid.total * throttle_pid.ki
            steer_pid.total += s
            control.steer = steer_pid.kp * s + steer_pid.total * steer_pid.ki+ (s-steer_pid.prev)*throttle_pid.kd
            steer_pid.prev = s
            print(control.steer, control.throttle)
        except:
            print("fir ruk gya")

        if control is None:
            print("kuch nhi")
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster')
    argparser.add_argument(
        '-m', '--map',
        action='store_true',
        help='plot the map of the current city')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                ############main function - execute
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
