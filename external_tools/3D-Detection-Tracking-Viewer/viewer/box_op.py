import numpy as np
from vedo import *

def convert_box_type(boxes,input_box_type = 'Kitti'):
    """
    convert the box type to unified box type
    :param boxes: (array(N,7)), input boxes
    :param input_box_type: (str), input box type
    :return: new boxes with box type [x,y,z,l,w,h,yaw]
    """
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return None
    assert  input_box_type in ["Kitti","OpenPCDet","Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet","Waymo"]:
        return boxes

    if input_box_type == "Kitti": #(h,w,l,x,y,z,yaw) -> (x,y,z,l,w,h,yaw)
        boxes = np.array(boxes)
        new_boxes = np.zeros(shape=boxes.shape)
        new_boxes[:,:]=boxes[:,:]
        new_boxes[:,0:3] = boxes[:,3:6]
        new_boxes[:, 3] = boxes[:, 2]
        new_boxes[:, 4] = boxes[:, 1]
        new_boxes[:, 5] = boxes[:, 0]
        new_boxes[:, 6] = (np.pi - boxes[:, 6]) + np.pi / 2
        new_boxes[:, 2] += boxes[:, 0] / 2
        return new_boxes

def get_mesh_boxes(boxes,colors="red",
                   mesh_alpha=0.4,
                   ids=None,
                   show_ids=False,
                   box_info=None,
                   show_box_info=False,
                   caption_size=(0.05,0.05)):
    """
    convert boxes array to vtk mesh boxes actors
    :param boxes: (array(N,7)), unified boxes array
    :param colors: (str or array(N,3)), boxes colors
    :param mesh_alpha: boxes transparency
    :param ids: list(N,), the ID of each box
    :param show_ids: (bool), show object ids in the 3D scene
    :param box_info: (list(N,)), a list of str, the infos of boxes to show
    :param show_box_info: (bool)，show object infos in the 3D Scene
    :return: (list(N,)), a list of vtk mesh boxes
    """
    vtk_boxes_list = []
    for i in range(len(boxes)):
        box = boxes[i]
        angle = box[6]

        new_angle = (angle / np.pi) * 180

        if type(colors) is str:
            this_c = colors
        else:
            this_c = colors[i]
        vtk_box = Box(pos=(0, 0, 0), height=box[5], width=box[4], length=box[3], c=this_c, alpha=mesh_alpha)
        vtk_box.rotateZ(new_angle)
        vtk_box.pos(box[0], box[1], box[2])

        info = ""
        if ids is not None and show_ids :
            info = "ID: "+str(ids[i])+'\n'
        if box_info is not None and show_box_info:
            info+=str(box_info[i])
        if info !='':
            vtk_box.caption(info,point=(box[0],
                            box[1]-box[4]/4, box[2]+box[5]/2),
                            size=caption_size,
                            alpha=1,c=this_c,
                            font="Calco",
                            justify='left')
            vtk_box._caption.SetBorder(False)
            vtk_box._caption.SetLeader(False)

        vtk_boxes_list.append(vtk_box)

    return vtk_boxes_list


def get_line_boxes(boxes,
                   colors,
                   show_corner_spheres=True,
                   corner_spheres_alpha=1,
                   corner_spheres_radius=0.3,
                   show_heading=True,
                   heading_scale=1,
                   show_lines=True,
                   line_width=2,
                   line_alpha=1,
                   ):
    """
    get vtk line box actors
    :param boxes: (array(N,7)), unified boxes array
    :param colors: (str or array(N,3)), boxes colors
    :param show_corner_spheres: (bool), show the corner points of box
    :param corner_spheres_alpha: (float), the transparency of corner spheres
    :param corner_spheres_radius: (float), the radius of of corner spheres
    :param show_heading: (bool), show the box heading
    :param heading_scale: (float), the arrow size of heading
    :param show_lines: (bool), show the lines of box
    :param line_width: (float), line width
    :param line_alpha: (float), line transparency
    :return: (list), a list of lines, arrows and spheres of vtk actors
    """
    lines_actors = []
    sphere_actors = []
    arraw_actors = []


    for i in range(len(boxes)):
        box = boxes[i]
        corner_points = []
        corner_points1 = []
        corner_points2 = []
        arraw_points1 = []
        arraw_points2 = []

        angle = box[6]

        new_angle = angle

        transform_mat = np.array([[np.cos(new_angle), -np.sin(new_angle), 0, box[0]],
                                  [np.sin(new_angle), np.cos(new_angle), 0, box[1]],
                                  [0, 0, 1, box[2]],
                                  [0, 0, 0, 1]])
        x = box[3]
        y = box[4]
        z = box[5]

        corner_points.append([-x / 2, -y / 2, -z / 2, 1])
        corner_points.append([-x / 2, -y / 2, z / 2, 1])
        corner_points.append([-x / 2, y / 2, z / 2, 1])
        corner_points.append([-x / 2, y / 2, -z / 2, 1])
        corner_points.append([x / 2, y / 2, -z / 2, 1])
        corner_points.append([x / 2, y / 2, z / 2, 1])
        corner_points.append([x / 2, -y / 2, z / 2, 1])
        corner_points.append([x / 2, -y / 2, -z / 2, 1])

        corner_points1.append([-x / 2, -y / 2, - z / 2, 1])
        corner_points1.append([-x / 2, -y / 2, z / 2, 1])
        corner_points1.append([-x / 2, y / 2, z / 2, 1])
        corner_points1.append([-x / 2, y / 2, -z / 2, 1])
        corner_points1.append([-x / 2, y / 2, z / 2, 1])
        corner_points1.append([-x / 2, -y / 2, z / 2, 1])
        corner_points1.append([x / 2, -y / 2, z / 2, 1])
        corner_points1.append([x / 2, y / 2, z / 2, 1])
        corner_points1.append([-x / 2, -y / 2, -z / 2, 1])
        corner_points1.append([x / 2, -y / 2, -z / 2, 1])
        corner_points1.append([x / 2, -y / 2, z / 2, 1])
        corner_points1.append([-x / 2, -y / 2, z / 2, 1])

        corner_points2.append([x / 2, -y / 2, - z / 2, 1])
        corner_points2.append([x / 2, -y / 2, z / 2, 1])
        corner_points2.append([x / 2, y / 2, z / 2, 1])
        corner_points2.append([x / 2, y / 2, -z / 2, 1])
        corner_points2.append([-x / 2, y / 2, -z / 2, 1])
        corner_points2.append([-x / 2, -y / 2, -z / 2, 1])
        corner_points2.append([x / 2, -y / 2, -z / 2, 1])
        corner_points2.append([x / 2, y / 2, -z / 2, 1])
        corner_points2.append([-x / 2, y / 2, -z / 2, 1])
        corner_points2.append([x / 2, y / 2, -z / 2, 1])
        corner_points2.append([x / 2, y / 2, z / 2, 1])
        corner_points2.append([-x / 2, y / 2, z / 2, 1])

        arraw_points1.append([0, 0, 0, 1])
        arraw_points2.append([x / 2 , 0, 0, 1])

        corner_points = np.matmul(np.array(corner_points), transform_mat.T)
        corner_points1 = np.matmul(np.array(corner_points1), transform_mat.T)
        corner_points2 = np.matmul(np.array(corner_points2), transform_mat.T)
        arraw_points1 = np.matmul(np.array(arraw_points1), transform_mat.T)
        arraw_points2 = np.matmul(np.array(arraw_points2), transform_mat.T)

        if type(colors) is not str:
            this_c = colors[i]
            corner_colors = np.tile(this_c,(corner_points.shape[0],1))
            arraw_colors = np.tile(this_c,(arraw_points1.shape[0],1))

        else:
            this_c = colors
            corner_colors = colors
            arraw_colors = colors

        lines = Lines(corner_points1[:, 0:3], corner_points2[:, 0:3], c=this_c, alpha=line_alpha, lw=line_width)

        corner_spheres = Spheres(corner_points[:,0:3], c= corner_colors, r=corner_spheres_radius,res = 12,alpha=corner_spheres_alpha)

        arraws = Arrows(arraw_points1[:,0:3],arraw_points2[:,0:3],c = arraw_colors,s=heading_scale)
        lines_actors.append(lines)
        sphere_actors.append(corner_spheres)
        arraw_actors.append(arraws)

    return_list =[]

    if show_corner_spheres:
        return_list+=sphere_actors
    if show_heading:
        return_list+=arraw_actors
    if show_lines:
        return_list+=lines_actors

    return return_list

def get_box_points(points, pose=None,show_box_heading=True):
    """
    box to points
    :param points: (7,),box
    :param pose:
    :return:
    """
    PI=np.pi
    import math
    point=np.zeros(shape=points.shape)
    point[:]=points[:]

    h,w,l = point[5],point[4],point[3]
    x,y,z = point[0],point[1],point[2]


    point_num=200
    i=1
    label=1
    z_vector = np.arange(- h / 2, h / 2, h / point_num)[0:point_num]
    w_vector = np.arange(- w / 2, w / 2, w / point_num)[0:point_num]
    l_vector = np.arange(- l / 2, l / 2, l / point_num)[0:point_num]

    d_z_p = -np.sort(-np.arange(0, h / 2, h / (point_num*2))[0:point_num])
    d_z_n = np.arange( -h / 2,0, h / (point_num*2))[0:point_num]


    d_w_p = -np.sort(-np.arange(0, w / 2, w / (point_num*2))[0:point_num])
    d_w_n = np.arange(-w / 2,0,  w / (point_num*2))[0:point_num]

    d_l_p = np.arange(l / 2, l*(4/7) , (l*(4/7)-l / 2) / (point_num*2))[0:point_num]


    d1 = np.zeros(shape=(point_num, 4))
    d1[:, 0] = d_w_p
    d1[:, 1] = d_l_p
    d1[:, 2] = d_z_p
    d1[:, 3] = i

    d2 = np.zeros(shape=(point_num, 4))
    d2[:, 0] = d_w_n
    d2[:, 1] = d_l_p
    d2[:, 2] = d_z_p
    d2[:, 3] = i

    d3 = np.zeros(shape=(point_num, 4))
    d3[:, 0] = d_w_p
    d3[:, 1] = d_l_p
    d3[:, 2] = d_z_n
    d3[:, 3] = i

    d4 = np.zeros(shape=(point_num, 4))
    d4[:, 0] = d_w_n
    d4[:, 1] = d_l_p
    d4[:, 2] = d_z_n
    d4[:, 3] = i

    z1 = np.zeros(shape=(point_num, 4))
    z1[:, 0] = -w / 2
    z1[:, 1] = -l / 2
    z1[:, 2] = z_vector
    z1[:, 3] = i
    z2 = np.zeros(shape=(point_num, 4))
    z2[:, 0] = -w / 2
    z2[:, 1] =l / 2
    z2[:, 2] = z_vector
    z2[:, 3] = i
    z3 = np.zeros(shape=(point_num, 4))
    z3[:, 0] = w / 2
    z3[:, 1] = -l / 2
    z3[:, 2] = z_vector
    z3[:, 3] = i
    z4 = np.zeros(shape=(point_num, 4))
    z4[:, 0] = w / 2
    z4[:, 1] = l / 2
    z4[:, 2] = z_vector
    z4[:, 3] = i
    w1 = np.zeros(shape=(point_num, 4))
    w1[:, 0]=w_vector
    w1[:, 1]=-l / 2
    w1[:, 2]=-h / 2
    w1[:, 3] = i
    w2 = np.zeros(shape=(point_num, 4))
    w2[:, 0] = w_vector
    w2[:, 1] = -l/ 2
    w2[:, 2] = h / 2
    w2[:, 3] = i
    w3 = np.zeros(shape=(point_num, 4))
    w3[:, 0] = w_vector
    w3[:, 1] = l / 2
    w3[:, 2] = -h / 2
    w3[:, 3] = i
    w4 = np.zeros(shape=(point_num, 4))
    w4[:, 0] = w_vector
    w4[:, 1] =l / 2
    w4[:, 2] = h / 2
    w4[:, 3] = i
    l1 = np.zeros(shape=(point_num, 4))
    l1[:, 0] = -w / 2
    l1[:, 1] = l_vector
    l1[:, 2] = -h / 2
    l1[:, 3] = i
    l2 = np.zeros(shape=(point_num, 4))
    l2[:, 0] = -w / 2
    l2[:, 1] = l_vector
    l2[:, 2] = h / 2
    l2[:, 3] = i
    l3 = np.zeros(shape=(point_num, 4))
    l3[:, 0] = w / 2
    l3[:, 1] = l_vector
    l3[:, 2] = -h / 2
    l3[:, 3] = i
    l4 = np.zeros(shape=(point_num, 4))
    l4[:, 0] = w / 2
    l4[:, 1] = l_vector
    l4[:, 2] = h / 2
    l4[:, 3] = i

    if show_box_heading:
        point_mat = np.asmatrix(np.concatenate((z1, z2, z3, z4, w1, w2, w3, w4, l1, l2, l3, l4, d1, d2, d3, d4)))
    else:
        point_mat = np.asmatrix(np.concatenate((z1, z2, z3, z4, w1, w2, w3, w4, l1, l2, l3, l4)))

    angle=point[6]-PI/2

    if pose is None:
        convert_mat = np.asmatrix([[math.cos(angle), -math.sin(angle), 0, x],
                              [math.sin(angle), math.cos(angle), 0, y],
                              [0, 0, 1, z],
                              [0, 0, 0, label]])

        transformed_mat = convert_mat * point_mat.T
    else:

        convert_mat = np.asmatrix([[math.cos(angle), -math.sin(angle), 0, 0],
                              [math.sin(angle), math.cos(angle), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        transformed_mat = convert_mat * point_mat.T
        pose_mat = np.asmatrix([[pose[0, 0], pose[0, 1], pose[0, 2], x],
                           [pose[1, 0], pose[1, 1], pose[1, 2], y],
                           [pose[2, 0], pose[2, 1], pose[2, 2], z],
                           [0, 0, 0, label]])
        transformed_mat = pose_mat * transformed_mat


    transformed_mat = np.array(transformed_mat.T,dtype=np.float32)

    return transformed_mat

def velo_to_cam(cloud,vtc_mat):
    """
    description: convert Lidar 3D coordinates to 3D camera coordinates .
    input: (PointsNum,3)
    output: (PointsNum,3)
    """
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.asmatrix(mat)
    normal=np.asmatrix(vtc_mat)
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T
