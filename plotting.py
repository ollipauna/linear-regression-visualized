import tkinter as tk
import numpy as np
from numpy.linalg import inv

# Simple App For Visualizing MSE Linear Regression


def left_click(event):
  global y, x, width, height, line_id, text_id
  r = 5

  if (line_id):
    canvas.delete(line_id)
    canvas.delete(text_id)
    pass

  y.append(event.y)
  x.append(event.x)
  point = canvas.create_oval(event.x+r, event.y+r, event.x-r, event.y-r, fill='red')
  points.append(point)

  N = len(x)

  if(N > 1):
    # Matrix of explanatory variables
    X = np.concatenate((np.transpose([np.ones(N)]), np.transpose([x])), axis=1)
    
    # Solve slope and bias
    XTX = np.matmul(X.T, X)
    XTX_INV = inv(XTX)
    w = np.matmul(np.matmul(y, X), XTX_INV)

    x0 = 0
    y0 = w[1]*x0 + w[0]

    x1 = width
    y1 = w[1]*x1 + w[0]

    line_id = canvas.create_line(x0, y0, x1, y1, fill='blue', width=5)
     
    # Display the slope and bias in a way that is intuitive for users
    slope = -round(w[1], 1)
    bias = round(height - w[0], 1)
    
    text_id = canvas.create_text(width*4/5, height*4/5, text=f"Slope: {slope} Bias: {bias}", fill="black", font=('Helvetica 16'))

def reset():
  global x, y, line_id, text_id, point
  
  canvas.delete(line_id)
  canvas.delete(text_id)
  for point in points:
    canvas.delete(point)
  
  y =  []
  x = []


if __name__ == '__main__':
  # Global vars
  width = 1600
  height = 900
  y = []
  x = []
  points = []
  line_id = None
  text_id = None

  root = tk.Tk()
  root.title('Linear Regression')
  
  canvas = tk.Canvas(root, width=width, height=height, bg='white')
  canvas.pack()
  canvas.bind('<Button-1>', left_click)
  
  B = tk.Button(canvas, text ="reset", command = reset, font=('Helvetica 16'))
  B.place(x=4/5*width, y=5/6*height)

  root.mainloop()