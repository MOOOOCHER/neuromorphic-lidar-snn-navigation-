#!/usr/bin/env python
import glob
import random
import sys

import pygame
from matplotlib import gridspec

sys.path.append('/usr/lib/python2.7/dist-packages')  # weil ROS nicht mit Anaconda installiert
# python2.7.17 path:  /home/learning/opt/python-2.7.17/bin/python
# import rospy

import math
import time
import numpy as np
import matplotlib.pyplot as plt
# import sim
# from collections import deque
import os.path
import tensorflow as tf
import fcn8_LoDNN_keras
import fcn_snn

from sklearn.cluster import DBSCAN
import cv2

from parameters import *

import carla


SECONDS_PER_EPISODE = 30


def normpdf(x, mean=0, sd=0.15):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


class CarlaEnvironment():
	def __init__(self, generateLidarLabels, useFCN, isTraining, scenario=""):
		self.generateLidarLabels = generateLidarLabels
		self.useFCN = useFCN
		self.isTraining = isTraining
		self.scenario = scenario #scenario1 or scenario2

		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(30)
		self.client.load_world('Town02')
		self.world = self.client.get_world()
		settings = self.world.get_settings()
		settings.fixed_delta_seconds = 0.4
		# settings.synchronous_mode = True
		self.world.apply_settings(settings)
		self.blueprint_library = self.world.get_blueprint_library()
		self.audi_2 = random.choice(self.blueprint_library.filter("vehicle.audi.a2"))
		# self.model_3 = self.blueprint_library.filter("model3")[0]
		self.buf = {'pts': np.zeros((1, 3)), 'intensity': np.zeros(1)}

		for actor in self.world.get_actors():
			if "vehicle." in  actor.type_id:
				actor.destroy()

		# -----  FCN Init  -----
		if self.useFCN:
			self.num_classes = 2
			self.image_shape = (200,400)
			self.lidar_im = None
			#self.fcn = fcn8_LoDNN_keras.LoDNN()
			self.fcn = fcn_snn.LoDNN_SNN()

		# -----  FCN Init  -----

		self.distance = 0
		self.steps = 0
		self.last_pos = None
		self.distance_traveled = 0
		self.v_pre = v_pre
		self.turn_pre = turn_pre
		self.start_innerLane = True

		self.trans_list = []
		self.count = 0

	def footage_init(self):
		self.camera_width = 1400
		self.camera_height = 480
		self.surface = None
		pygame.init()
		self.display = pygame.display.set_mode((self.camera_width, self.camera_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
		self.display.fill((255, 255, 255))
		pygame.display.flip()

	def footage_camera_init(self):
		camera = self.blueprint_library.find('sensor.camera.rgb')
		camera.set_attribute('image_size_x', str(self.camera_width))
		camera.set_attribute('image_size_y', str(self.camera_height))
		self.camera = self.world.spawn_actor(camera,
											 carla.Transform(carla.Location(x=0.0, z=1.7), carla.Rotation(pitch=0.0)),
											 attach_to=self.vehicle,
											 attachment_type=carla.AttachmentType.Rigid)
		self.actor_list.append(self.camera)
		self.camera.listen(lambda image: self.camera_callback(image))

	def plot_init(self):
		########################STATE##################################
		plt.ion()
		self.fig = plt.figure(figsize=(2, 4.5))
		spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])
		ax1 = self.fig.add_subplot(spec[0])
		plt.title('Lane Feature')
		self.state_plot = ax1.imshow(np.zeros((resolution[1], resolution[0])), alpha=1, cmap='PuBu', vmax=1, aspect='equal')
		plt.axis('off')

		self.motor_ax2 = self.fig.add_subplot(spec[1])
		plt.title('Actuator Output')
		self.motor_plot = self.motor_ax2.imshow(np.zeros((2, 1)), alpha=1, cmap='PuBu', vmax=1, aspect='auto')
		self.motor_l_value_plot = self.motor_ax2.text(-0.25, 1.0, 0, ha='center', va='center')
		self.motor_r_value_plot = self.motor_ax2.text(0.25, 1.0, 0, ha='center', va='center')
		plt.axis('off')

		self.err_fig = plt.figure(figsize=(4, 4))
		ax_err = self.err_fig.add_subplot(111)
		ax_err.set_ylabel('Lane Deviation [m]')
		ax_err.set_xlabel('Testing Step')
		ax_err.set_xlim((0, 1500))
		ax_err.set_ylim((-2, 2))
		self.d_err = []
		self.err_plot, = ax_err.plot([0, 0])

	def reset(self):
		self.collision_hist = []
		self.actor_list = []

		if self.isTraining:
			if self.start_innerLane:
				location_vehicle = carla.Location(165.0, 191.64, 0.5) # 6
				rotation_vehicle = carla.Rotation(0, 0, 0)
			else:
				location_vehicle = carla.Location(165.0, 187.8, 0.5)  # 6
				rotation_vehicle = carla.Rotation(0, 180, 0)
			self.start_innerLane = self.start_innerLane ^ True
		else:
			location_vehicle = carla.Location(165.0, 191.64, 0.5)  # 6
			rotation_vehicle = carla.Rotation(0, 0, 0)

		self.transform = carla.Transform(location_vehicle, rotation_vehicle)
		self.vehicle = self.world.spawn_actor(self.audi_2, self.transform)

		self.actor_list.append(self.vehicle)

		self.lidar_se = self.blueprint_library.find('sensor.lidar.ray_cast_semantic')
		## Set the parameters of semantic lidar
		self.lidar_se.set_attribute('channels', '32')
		self.lidar_se.set_attribute('points_per_second', '200000')
		self.lidar_se.set_attribute('upper_fov', '-1')  # 0
		self.lidar_se.set_attribute('lower_fov', '-20.0')  # -20
		self.lidar_se.set_attribute('range', '30')  # Maximum distance
		self.lidar_se.set_attribute('rotation_frequency', '20')
		self.lidar_se.set_attribute('sensor_tick', '0.2')  # 1/frequency: 1/10
		## Set the position of the Lidar relative to the attached object
		transform_lidar_se = carla.Transform(carla.Location(x=0, y=0, z=1.7))  # x=0, y=0, z=1.5
		self.vehicle_lidar_se = self.world.spawn_actor(self.lidar_se, transform_lidar_se, attach_to=self.vehicle)
		self.actor_list.append(self.vehicle_lidar_se)

		if self.useFCN or self.generateLidarLabels:
			self.vehicle_lidar_se.listen(lambda data: self.semantic_lidar_callback(data, self.buf))

		colsensor = self.blueprint_library.find("sensor.other.collision")
		transform = carla.Transform(carla.Location(x=2.5, z=0.7))
		self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		# while self.lidar_im is None:
		# 	time.sleep(0.01)

		self.episode_start = time.time()
		# reset speed as 0
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		time.sleep(1)
		return np.zeros((resolution[0], resolution[1]), dtype=int), 0.    # state; reward

	def collision_data(self, event):
		self.collision_hist.append(event)

	def semantic_lidar_callback(self, point_cloud, buf):
		"""Prepares a point cloud with semantic segmentation colors"""
		data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
			('x', np.float32), ('y', np.float32), ('z', np.float32),
			('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

		# We're negating the y to correclty visualize a world that matches
		# what we see in Unreal since Open3D uses a right-handed coordinate system
		points = np.array([data['x'], -data['y'], data['z']]).T

		self.trans_list.append(self.vehicle.get_transform())
		if len(self.trans_list) > 7:
			cur_trans = self.trans_list[-7]
		else:
			cur_trans = self.trans_list[-1]
		cur_pos = cur_trans.location
		cur_rot = -1.0 * cur_trans.rotation.yaw / 180.0 * math.pi
		wpt = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)

		if self.isTraining:
			road_ids = [14, 68, 10, 157, 9, 322, 6, 283, 317, 148, 62, 290]  # loop- clockwise and counter clockwise for training
		else:
			if self.scenario == "scenario2":
				road_ids = [14, 68, 10, 157, 9, 6, 283, 308, 5, 224, 8, 131, 11]  # 8-loop roads
			else:
				road_ids = [14, 68, 10, 157, 9, 322, 6, 283, 317, 148, 62, 290]

		road_width = wpt.lane_width
		waypoints = self.world.get_map().generate_waypoints(0.1)

		# Colorize the pointcloud based on the CityScapes color palette
		labels = np.array(data['ObjTag'], dtype=np.float32)
		# int_color = labels

		# # In case you want to make the color intensity depending
		# # of the incident ray angle, you can use:
		#  int_color *= np.array(data['CosAngle'])

		buf['pts'] = points
		buf['intensity'] = labels

		lidar_points = np.column_stack((buf['pts'], buf['intensity'].reshape(-1, 1)))

		# data process
		x_points = lidar_points[:, 0]
		y_points = lidar_points[:, 1]
		# l_points = lidar_data[:, 3]  # labels

		side_range = (-10, 10)  # left-most to right-most
		fwd_range = (0, 20)  # back-most to forward-most  (0, 40)

		# FILTER - To return only indices of points within desired cube
		# Three filters for: Front-to-back, side-to-side
		# Note left side is positive y axis in LIDAR coordinates
		x_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
		y_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
		filter = np.logical_and(x_filt, y_filt)
		indices = np.argwhere(filter).flatten()

		# KEEPERS
		x_points = x_points[indices]  # (4952,)
		y_points = y_points[indices]  # (4952,)

		# plt.scatter(x_points, y_points)
		# plt.show()

		res = 0.05
		# CONVERT TO PIXEL POSITION VALUES - Based on resolution
		x_img = (np.floor(x_points / res)).astype(np.int32)  # x axis is -y in LIDAR
		y_img = (y_points / res).astype(np.int32)  # y axis is -x in LIDAR

		# SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
		# floor and ceil used to prevent anything being rounded to below 0 after shift
		x_img -= int(np.floor(fwd_range[0] / res))
		y_img -= int(np.floor(side_range[0] / res))

		# INITIALIZE EMPTY ARRAY - of the dimensions we want
		x_max = int((fwd_range[1] - fwd_range[0]) / res)
		y_max = int((side_range[1] - side_range[0]) / res)
		self.lidar_im = np.zeros([x_max, y_max], dtype=np.uint8)  # initialize black picture

		# x_img = x_max - x_img
		y_img = y_max - y_img
		# FILL PIXEL VALUES IN IMAGE ARRAY
		self.lidar_im[x_img, y_img] = 255  # lidar points as white color
		self.lidar_im = np.flip(self.lidar_im, 0)
		if self.generateLidarLabels:
			x_bound = 20  # 8x4
			y_bound = 10  # 8x4

			rotated_lane = [[0, 0]]
			filtered_waypoints = []
			for waypoint in waypoints:
				if waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id in road_ids:
					trans = waypoint.transform
					o_x = trans.location.x - cur_pos.x
					o_y = trans.location.y - cur_pos.y
					x = o_x * math.cos(cur_rot) - o_y * math.sin(cur_rot)
					y = o_x * math.sin(cur_rot) + o_y * math.cos(cur_rot)
					if 0 <= x <= x_bound and -y_bound <= y <= y_bound:
						rotated_lane.append([x, y])
						filtered_waypoints.append(waypoint)
			rotated_lane = np.array(rotated_lane)
			clustering = DBSCAN(eps=reset_distance, min_samples=2).fit(rotated_lane)
			labels = clustering.labels_
			cur_path = labels[0]
			rotated_lane = rotated_lane[1:]
			labels = labels[1:]
			cur_path_indices = np.flatnonzero(labels == cur_path)
			rotated_lane = rotated_lane[cur_path_indices]
			img_buf = np.zeros((int(2 * y_bound / res), int(x_bound / res)), dtype=np.uint8)
			for point in rotated_lane:
				x = int(point[0] / res)
				y = int((point[1] + y_bound) / res)
				img_buf[y, x] = 255

			img_buf = cv2.dilate(img_buf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(road_width / res), int(road_width / res))),
								 borderType=cv2.BORDER_REPLICATE)
			img_buf_boundary = cv2.Laplacian(img_buf, cv2.CV_64F)
			img_buf_boundary = np.uint8(np.absolute(img_buf_boundary))
			img_buf_boundary = cv2.dilate(img_buf_boundary,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(0.3 / res), int(0.3 / res))),
								 borderType=cv2.BORDER_REPLICATE)
			# cv2.imshow("lane area", img_buf_rotated)
			# cv2.waitKey(1)
			img_buf_rotated_boundary = cv2.rotate(img_buf_boundary, cv2.ROTATE_90_COUNTERCLOCKWISE)
			img_buf_rotated_boundary = cv2.resize(img_buf_rotated_boundary, self.lidar_im.shape,
										 interpolation=cv2.INTER_AREA)
			lidar_label_boundary = np.dot((img_buf_rotated_boundary > 150).reshape(img_buf_rotated_boundary.shape[0], img_buf_rotated_boundary.shape[1], 1),
								 np.array([[0, 100, 0]]))

			if self.count%1==0:
				cv2.imwrite('../lidar_data/LidarLabels/New/Nengo/velodyne_bv_road/um_{0:06}.png'.format(self.count), self.lidar_im)
				cv2.imwrite('../lidar_data/LidarLabels/New/Nengo/gtbv2/um_road_{0:06}.png'.format(self.count), lidar_label_boundary)
			self.count = self.count + 1


	def plot_err(self):
		self.d_err.append([self.steps, self.distance])
		self.err_plot.set_xdata(np.array(self.d_err)[:, 0])
		self.err_plot.set_ydata(np.array(self.d_err)[:, 1])
		self.err_fig.canvas.draw()
		self.err_fig.canvas.flush_events()

	def plot_neurons(self, s, m_l, m_r):
		self.state_plot.set_data(np.flipud(s.T))
		self.motor_plot.set_data(np.array([[m_l, m_r]]))
		self.motor_l_value_plot.set_text('%.2f' % m_l)
		self.motor_r_value_plot.set_text('%.2f' % m_r)
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

	def plot_footage(self):
		pygame.event.get()
		self.display.blit(self.surface, (0, 0))
		pygame.display.flip()

	def step(self, n_l, n_r, episode):

		# top view
		spectator = self.world.get_spectator()
		transform_vec = self.vehicle.get_transform()
		spectator.set_transform(carla.Transform(transform_vec.location + carla.Location(z=40),
												carla.Rotation(pitch=-90)))

		self.steps += 1
		t = False  # initialize terminal state
		# print('l,r:', n_l, n_r)
		# Steering wheel model
		m_l = n_l/n_max
		m_r = n_r/n_max
		a = m_l - m_r
		v_cur = - abs(a)*(v_max - v_min) + v_max
		turn_cur = turn_factor * a
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.v_pre = c*v_cur + (1-c)*self.v_pre            # vehicle speed
		self.turn_pre = (turn_cur + self.turn_pre) / 2.0   # turn speed

		#print "m_l: {}, m_r: {}, turn: {}".format(m_l, m_r, self.turn_pre)

		# print(self.v_pre, self.turn_pre)   # self.v_pre: 1.5->1.0->..  self.turn_pre:0->-0.47->..
		# v_left = self.v_pre + self.turn_pre
		# v_right = self.v_pre - self.turn_pre
		self.vehicle.apply_control(carla.VehicleControl(throttle=self.v_pre, steer=self.turn_pre))
		# Minus - left turn;  Plus - right turn

		if self.useFCN:
			s = self.fcn.generateNewState(self.lidar_im)

		x_bound = 7  # 8x4
		y_bound = 5  # 8x4

		res = 0.1

		cur_trans = self.vehicle.get_transform()
		cur_pos = cur_trans.location

		if not self.last_pos:
			self.last_pos = cur_pos
		dx = cur_pos.x - self.last_pos.x
		dy = cur_pos.y - self.last_pos.y
		self.distance_traveled += math.sqrt(dx ** 2 + dy ** 2)
		self.last_pos = cur_pos

		cur_rot = -1.0 * cur_trans.rotation.yaw / 180.0 * math.pi
		wpt = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)

		if self.isTraining:
			road_ids = [14, 68, 10, 157, 9, 322, 6, 283, 317, 148, 62, 290]  # loop- clockwise and counter clockwise for training
		else:
			if self.scenario == "scenario2":
				road_ids = [14, 68, 10, 157, 9, 6, 283, 308, 5, 224, 8, 131, 11]  # 8-loop roads
			else:
				road_ids = [14, 68, 10, 157, 9, 322, 6, 283, 317, 148, 62, 290]

		road_width = wpt.lane_width
		waypoints = self.world.get_map().generate_waypoints(0.1)
		rotated_lane = [[0, 0]]
		filtered_waypoints = []
		for waypoint in waypoints:
			if waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id in road_ids:
				trans = waypoint.transform
				o_x = trans.location.x - cur_pos.x
				o_y = trans.location.y - cur_pos.y
				x = o_x * math.cos(cur_rot) - o_y * math.sin(cur_rot)
				y = o_x * math.sin(cur_rot) + o_y * math.cos(cur_rot)
				if 0 <= x <= x_bound and -y_bound <= y <= y_bound:
					rotated_lane.append([x, y])
					filtered_waypoints.append(waypoint)
		rotated_lane = np.array(rotated_lane)
		clustering = DBSCAN(eps=reset_distance, min_samples=2).fit(rotated_lane)
		labels = clustering.labels_
		cur_path = labels[0]
		rotated_lane = rotated_lane[1:]
		labels = labels[1:]
		cur_path_indices = np.flatnonzero(labels == cur_path)
		rotated_lane = rotated_lane[cur_path_indices]
		filtered_waypoints = np.array(filtered_waypoints)[cur_path_indices]

		if not self.useFCN:
			img_buf = np.zeros((int(2 * y_bound / res), int(x_bound / res)), dtype=np.uint8)
			for point in rotated_lane:
				x = int(point[0] / res)
				y = int((point[1] + y_bound) / res)
				img_buf[y, x] = 255

			img_buf = cv2.dilate(img_buf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
																	(int(road_width / res), int(road_width / res))),
								 borderType=cv2.BORDER_REPLICATE)
			img_buf_border = cv2.Laplacian(img_buf, cv2.CV_64F)
			img_buf_border = np.uint8(np.absolute(img_buf_border))
			# cv2.imshow("img", img_buf)
			# cv2.waitKey(1)
			new_state = cv2.resize(img_buf_border, (resolution[1], resolution[0]), interpolation=cv2.INTER_AREA)
			s = np.where(new_state > 0, 1, 0)

		print(s)

		n = self.steps


		# # Get location of the vehicle
		# location = self.vehicle.get_location()
		# # print(location.x, location.y, location.z, 'b')   # 'b' and 'c' the same
		# or
		veh_trans = self.vehicle.get_transform()
		veh_loc = veh_trans.location
		veh_rot = veh_trans.rotation

		d_delta = 0.0
		if len(rotated_lane):
			wpt = filtered_waypoints[np.argmin(rotated_lane, 0)[0]]
			wpt_loc = wpt.transform.location
			wpt_rot = wpt.transform.rotation

			# print("x dist: ", veh_loc.x - wpt_loc.x, "y dist: ", veh_loc.y - wpt_loc.y)
			yaw_rad = wpt_rot.yaw * math.pi / 180.0
			d = (wpt_loc.x - veh_loc.x) * math.sin(yaw_rad) - (
						wpt_loc.y - veh_loc.y) * math.cos(yaw_rad)
			veh_yaw = veh_rot.yaw * math.pi / 180.0
			#print("vehicle yaw: {}, waypoint yaw: {}".format(veh_rot.yaw, wpt_rot.yaw))
			d_delta = self.v_pre * math.tan(veh_yaw - yaw_rad)
			#print('d {}, delta d {}, v {}'.format(d, d_delta, self.v_pre))
		else:
			d = reset_distance + 0.1

		p = self.distance_traveled

		# Calculate reward
		r = 3.0 * d + 5.0 * d_delta

		self.distance = d

		if_reset = False
		steps = self.steps
		# Terminate episode if robot reaches the reset distance or collision
		if len(self.collision_hist) != 0 or abs(d) > reset_distance or n >= max_step:  #or self.episode_start + SECONDS_PER_EPISODE < time.time()
			s,r,t, if_reset = self.reset_step(n)
		# Terminate episode if robot reaches a certain position
		if not self.isTraining:
			if self.scenario == "scenario1":
				if 164 <= veh_loc.x <= 165 and 186<=veh_loc.y<=197:
				# Terminate scenario 1
					s,r,t, if_reset = self.reset_step(n)
			elif self.scenario =="scenario2":
				if 112 <= veh_loc.x <= 114 and 236<=veh_loc.y<=245:
				# Terminate scenario 2
					s,r,t, if_reset = self.reset_step(n)
		print('reward: {}, steps: {}, distance traveled this episode: {}'.format(r, steps, p))
		# Return state, distance, position, reward, termination, steps, vehicle location
		return s, d, p, r, t, n, steps, if_reset, veh_loc, self.v_pre

	def reset_step(self, n):
		if_reset = True
		self.steps = self.distance = 0
		self.distance_traveled = 0
		self.last_pos = None
		t = True
		print('steps:', n)
		for actor in self.actor_list:
			actor.destroy()
		s, r = self.reset()
		time.sleep(2)
		return s, r, t, if_reset
