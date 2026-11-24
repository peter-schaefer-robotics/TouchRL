import random
import math
import numpy as np

def draw_circle(grid_size):
    # draw a circle at the center position cx,cy with the radius r within the given grid size as a mask of zeros and ones
    # the circle has to be fully inside the grid (can be changed later to make it harder)
    r = random.randint(1, (grid_size - 1) // 2)
    cx = random.randint(r, grid_size - 1 - r)
    cy = random.randint(r, grid_size - 1 - r)
    ys, xs = np.indices((grid_size, grid_size))
    mask = (xs-cx)**2 + (ys-cy)**2 <= r**2
    return mask

def draw_rectangle(grid_size):
    # draw a rectangle at the base position bx, by (left upper field), with lengths lx,ly within the given grid size as a mask of zeros and ones
    # the rectangle has to be fully inside the grid (can be changed later to make it harder)
    lx = random.randint(1, grid_size)
    ly = random.randint(1, grid_size)
    bx = random.randint(0, grid_size - lx)
    by = random.randint(0, grid_size - ly)
    ys, xs = np.indices((grid_size, grid_size))
    mask = (xs >= bx) & (xs < bx + lx) & (ys >= by) & (ys < by + ly)
    return mask

def draw_triangle_isosceles(grid_size):
    # draw a isosceles triangle at bx, by = middle cell of the base (bottom row), tip points up, with base length lb within the given grid size as a mask of zeros and ones
    # base length lb must be odd to simplify it in this discrete space. height is per definition half lb rounded to ceiling
    # the triangle has to be fully inside the grid (can be changed later to make it harder)
    lb = random.randrange(3, grid_size + 1, 2)      # only odd values: 3, 5, 7, ...
    h = math.ceil(lb/2)
    bx = random.randint(math.floor(lb/2), grid_size - math.floor(lb/2) - 1)
    by = random.randint(h - 1, grid_size - 1)
    ys, xs = np.indices((grid_size, grid_size))
    top = by - h + 1                                # row of the tip
    mask = (ys >= top) & (ys <= by) & (np.abs(xs - bx) <= ys - top)
    return mask
       
SHAPES = {
    "circle" : draw_circle, # shape 0
    "rectangle" : draw_rectangle, # shape 1
    "triangle" : draw_triangle_isosceles # shape 2
}