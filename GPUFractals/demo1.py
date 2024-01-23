import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import numba as nb
import cupy as cp

import time

class FractalGenerator():
    def __init__(self,func,mode):
        self.func = func
        if (mode == 'cpu'):
            self.mode = 0
        elif(mode == 'gpu'):
            self.mode = 1
        else:
            print('Invalid mode')
            self.mode = -1
        self.genInvJac()

    def genInvJac(self):
        x = sp.Symbol('x', real=True)
        y = sp.Symbol('y', real=True)
        z = sp.Symbol('z')

        strExpr = sp.sympify(self.func)

        strExpr = strExpr.subs(z, x+1j*y)
        sp.Q.real(x)
        sp.Q.real(y)
        realExpr = sp.re(strExpr)
        imagExpr = sp.im(strExpr)

        self.V = sp.Matrix([realExpr,imagExpr])
        X = sp.Matrix([x,y])

        self.jacobian = self.V.jacobian(X)

        if (self.mode == 1):
            import cupy as cp
            self.fx = convert_power_arg_to_float64(sp.ccode(self.jacobian[0,0]))
            self.fy = convert_power_arg_to_float64(sp.ccode(self.jacobian[0,1]))
            self.gx = convert_power_arg_to_float64(sp.ccode(self.jacobian[1,0]))
            self.gy = convert_power_arg_to_float64(sp.ccode(self.jacobian[1,1]))

            self.Vx = convert_power_arg_to_float64(sp.ccode(self.V[0]))
            self.Vy = convert_power_arg_to_float64(sp.ccode(self.V[1]))
            self.x = x
            self.y = y

        elif(self.mode == 0):
            self.invJac = self.jacobian.inv()
            self.jac = sp.lambdify(X,self.jacobian,'numpy')
            self.f_system = sp.lambdify(X,self.V,'numpy')
        else:
            pass

    WHITE = np.array([1.0,1.0,1.0,1.0])
    RED = np.array([1.0,0.0,0.0,1.0])
    GREEN = np.array([0.0,0.1,0.0,1.0])
    BLUE = np.array([0.0,0.0,1.0,1.0])
    PURPLE = np.array([0.8,0.0,0.8,1.0])
    ORANGE = np.array([0.8,0.4,0.0,1.0])
    YELLOW = np.array([0.8,0.6,0.0,1.0])
    LIME = np.array([4.0,0.8,0.0,1.0])
    MAGENTA = np.array([0.8,0.0,0.4,1.0])

    COLORLIST = [WHITE,RED,GREEN,BLUE,PURPLE,ORANGE,YELLOW,LIME,MAGENTA]

    def runNR(self,x,y,tol,maxIters):


        if (self.mode == 1):
            self.runNRGPU(x,y,tol,maxIters)
        if (self.mode == 0):
            self.runNR_numba(x,y,tol,maxIters)

    def runNRGPU(self, x, y, tol, maxIters):
        self.x = cp.array(cp.array(x.flatten(), dtype=np.float64))
        self.y = cp.array(cp.array(y.flatten(), dtype=np.float64))
        KernelInput = 'float64 x0, float64 y0, float64 tol'

        KernelOutput = 'float64 x, float64 y, float64 convergence, bool ongoing'

        KernelA = '''
        x = x0;
        y = y0;

        '''

        fxString = '\ndouble fx = ' + self.fx + ';\n'

        fyString = '\ndouble fy = ' + self.fy + ';\n'

        gxString = '\ndouble gx = ' + self.gx + ';\n'

        gyString = '\ndouble gy = ' + self.gy + ';\n'

        VxString = '\ndouble Vx = ' + self.Vx + ';\n'

        VyString = '\ndouble Vy = ' + self.Vy + ';\n'

        KernelB = '''

        double det = (fx*gy - fy*gx);

        double a = gy/det;
        double b = -fy/det;
        double c = -gx/det;
        double d = fx/det;

        double xStep = -a*Vx - b*Vy;
        double yStep = -c*Vx - d*Vy;

        x = x + xStep;
        y = y + yStep;

        convergence = xStep*xStep + yStep*yStep;

        ongoing = true;
        if (convergence < tol*tol){
            ongoing = false;
        }
        '''

        Kernel = KernelA + fxString + fyString + gxString + gyString + VxString + VyString + KernelB

        NR = cp.ElementwiseKernel(KernelInput, KernelOutput, Kernel, 'NewtonRaphson')

        xi = self.x.copy()
        yi = self.y.copy()
        convergence = self.x*0
        self.ItersMap = self.x*0

        for i in range(0, maxIters):
            xi, yi, convergence, ongoing = NR(xi, yi, tol)
            self.ItersMap = self.ItersMap + ongoing*1

        self.xMap = cp.asnumpy(xi)
        self.yMap = cp.asnumpy(yi)
        self.ItersMap = cp.asnumpy(self.ItersMap)
        self.ConvMap = cp.asnumpy(convergence**(1/2))

    def runNR_numba(self,x,y,tol,maxIters):

        jac = self.jac
        f_system = self.f_system

        xMap = x*0
        yMap = x*0
        itersMap = x*0
        convMap = x*0

        m,n = np.shape(x)

        print(y)

        for i in range(0,m):
            for j in range(0,n):
                tolerance = tol

                iters = 0

                xGuess = np.squeeze(x[i,j])
                yGuess = np.squeeze(y[i,j])

                while (iters < maxIters):

                    if (tolerance < tol):
                        itersMap[i,j] -= 1

                    Jinv_system = np.linalg.inv(jac(xGuess,yGuess))

                    xStep = -Jinv_system[0,0]*f_system(xGuess,yGuess)[0] - Jinv_system[0,1]*f_system(xGuess,yGuess)[1]
                    yStep = -Jinv_system[1,0]*f_system(xGuess,yGuess)[0] - Jinv_system[1,1]*f_system(xGuess,yGuess)[1]

                    xGuess = np.squeeze(xStep + xGuess)
                    yGuess = np.squeeze(yStep + yGuess)

                    tolerance = np.sqrt(xStep**2 + yStep**2)

                    itersMap[i,j] += 1

                    iters = iters + 1

                xMap[i,j] = xGuess
                yMap[i,j] = yGuess
                convMap[i,j] = tolerance


        self.xMap = xMap
        self.yMap = yMap
        self.ItersMap = itersMap-1
        self.ConvMap = convMap

def swap_power_arguments(inputt):

    try:
        inputt.lower()
    except:
        print('not a string')

    stringy = []

    for i in range(0,len(inputt)):
        stringy.append(inputt[i])

    for i in range(0,len(stringy)-3):
        if (stringy[i] == 'p' and stringy[i+1] == 'o' and stringy[i+2] == 'w'):
            j = i+3
            mid = i+3
            arglength = 0
            while(stringy[j] != ')'):
                j += 1
                arglength += 1
                if(stringy[j] == ','):
                    mid = j

            firstarg = stringy[i+4:mid]
            secondarg = stringy[(mid+1):(i+3+arglength)]
            stringy[i+4:(i+4+len(secondarg))] = secondarg
            stringy[(i+4+len(secondarg))] = ','
            stringy[(i+4+len(secondarg)+1):(i+4+len(secondarg)+1)+len(firstarg)] = firstarg

    string = ''

    for i in range(0,len(stringy)):
        string += stringy[i]

    return string

def convert_power_arg_to_float64(inputt):

    try:
        inputt.lower()
    except:
        print('not a string')

    stringy = []

    for i in range(0,len(inputt)):
        stringy.append(inputt[i])

    for i in range(0,len(stringy)-3):
        if (stringy[i] == 'p' and stringy[i+1] == 'o' and stringy[i+2] == 'w'):
            j = i+3
            while(stringy[j] != ')'):
                j += 1
            stringy[j-1]=stringy[j-1]+'.0'

    string = ''

    for i in range(0,len(stringy)):
        string += stringy[i]

    return string


#%% Control
rando = np.random.rand()

m = 801
n = 801

x = np.linspace(-2,2,m)
y = np.linspace(-2,2,n)
y,x = np.meshgrid(y,x)
tol = 0.0005
maxIters = 50

func = 'cos(z*10) + z**3 + 1.5*z**2 - 2 + 2j'
func = 'z**3 + cos(z**2) - 1 + 3j'
# FG = FractalGenerator(func,'cpu')
# FG.runNR(x,y,tol,maxIters)
# convmap1 = FG.ItersMap

# pg.image(convmap1)


time1 = time.time()
FG = FractalGenerator(func,'gpu')
FG.runNR(x,y,tol,maxIters)
convmap2 = np.reshape(FG.ItersMap,[m,n])

print(time.time()-time1)

plt.pcolor(convmap2)


