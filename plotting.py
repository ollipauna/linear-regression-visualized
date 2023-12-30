import tkinter as tk
import numpy as np
from numpy.linalg import inv

# Simple App For Visualizing MSE Linear Regression

def standardize(data):
   mu = np.mean(data)
   sigma = np.std(data)
   return (data - mu) / sigma, mu, sigma

def scale(point, mu, sigma):
   return (point - mu) / sigma

def scale_back(data, mu, sigma):
   return data * sigma + mu

def closed_form(N):
    global y, x, width, height, line_id, text_id  
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
    slope = -round(w[1], 2)
    bias = round(height - w[0], 2)
    
    text_id = canvas.create_text(width*4/5, height*4/5, text=f"Closed Form Slope: {slope} Bias: {bias}", fill="black", font=('Helvetica 16'))

def gradient_descent(N):
    global y, x, width, height, line_id2, text_id2
    learning_rate = 0.01
    epochs = 20

    x_norm, x_mu, x_sigma = standardize(x)
    y_norm, y_mu, y_sigma = standardize(y)

    a = 0
    b = scale(height/2, y_mu, y_sigma)


    for epoch in range(epochs):
        delta_a = 0
        delta_b = 0

        for i in range(N):
          delta_a += learning_rate * x_norm[i] * (a * x_norm[i] + b - y_norm[i])
          delta_b += learning_rate *(a * x_norm[i] + b - y_norm[i])
        
        a -= delta_a
        b -= delta_b

    
    x0 = scale(0, x_mu, x_sigma)
    y0 = a*x0 + b 
    y0 = scale_back(y0, y_mu, y_sigma)

    x1 = scale(width, x_mu, x_sigma)
    y1 = a*x1 + b 
    y1 = scale_back(y1, y_mu, y_sigma)

    line_id2 = canvas.create_line(0, y0, width, y1, fill='red', width=5)
     
    # Display the slope and bias in a way that is intuitive for users
    slope = round(-(y1- y0)/(width - 0), 2)
    bias = round(height - y0, 2)
    
    text_id2 = canvas.create_text(width*1/5, height*1/5, text=f"Gradient Descent Slope: {slope} Bias: {bias}", fill="black", font=('Helvetica 16'))


def clear_canvas():
   global line_id, line_id2, text_id, text_id2

   if (line_id):
    canvas.delete(line_id)
    canvas.delete(line_id2)
    canvas.delete(text_id)
    canvas.delete(text_id2)
      

def left_click(event):
  global y, x, width, height
  r = 5

  clear_canvas()

  y.append(event.y)
  x.append(event.x)
  point = canvas.create_oval(event.x+r, event.y+r, event.x-r, event.y-r, fill='red')
  points.append(point)

  N = len(x)

  if(N > 1):
    closed_form(N)
    gradient_descent(N)

def reset():
  global x, y, point
  
  clear_canvas()
  
  for point in points:
    canvas.delete(point)
  
  y =  []
  x = []


if __name__ == '__main__':
  # Global vars
  width = 1400
  height = 900
  y = []
  x = []
  points = []
  line_id = None
  line_id2 = None
  text_id = None
  text_id2 = None

  root = tk.Tk()
  root.title('Linear Regression')
  
  canvas = tk.Canvas(root, width=width, height=height, bg='white')
  canvas.pack()
  canvas.bind('<Button-1>', left_click)
  
  B = tk.Button(canvas, text ="reset", command = reset, font=('Helvetica 16'))
  B.place(x=4/5*width, y=5/6*height)

  root.mainloop()