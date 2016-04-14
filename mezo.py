import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
from pylab import *
from numpy import *
from numpy.linalg import *
import scipy.linalg as sl # we need a generailzed eigen solver


class lead1D:
    'A class for simple 1D leads'
    def __init__(self,eps0=0,gamma=-1,**kwargs):
        'We assume real hopping \gamma and onsite \epsilon_0 parameters!'
        self.eps0=eps0
        self.gamma=gamma
        return
    
    def Ek(self,k):
        'Spectrum as a function of k'
        return self.eps0+2*self.gamma*cos(k)
    
    def kE(self,E,**kwargs):
        '''
        Spectrum as a function of E.
        If keyword a=True is given than 
        it gives back two k values, 
        one positive and one negative.
        '''
        a = kwargs.get('a',False)
        k=arccos((E-self.eps0)/(2*self.gamma))
        if a:
            return array([-k,k])
        else:
            return k
        
    def vE(self,E=0,**kwargs):
        '''
        Group velocity as a function of E.
        If keyword a=True is given than 
        it gives back two v values, 
        one positive and one negative.
        '''
        a = kwargs.get('a',False)
        k=self.kE(E)
        v= -2*self.gamma*sin(k)
        if a:
            return array([-v,v])
        else:
            return v
    
    
    def sgf(self,E=0):
        '''
        Surgace Green's function of a seminfinite 1D lead.
        '''
        return exp(1.0j *self.kE(E))/self.gamma
    
    def sgfk(self,k=pi/2):
        '''
        Surgace Green's function of a seminfinite 1D lead in terms of k.
        '''
        return exp(1.0j*k)/self.gamma
    
    def vk(self,k=pi/2):
        '''
        Group velocity in terms of k
        '''
        return -2*self.gamma*sin(k)
        

class lead:
    '''
    This is a class for a generic lead.
    Without degenerate subspace ironing!
    Without 
    '''
    def __init__(self,H0,H1,**kwargs):
        '''
        Initialization of the leads.
        Maybe some checks could come here later on
        or even the svd...
        '''
        self.H0=matrix(H0)
        self.H1=matrix(H1)
        self.dim=shape(H0)[0]
        
        self.tol=1.0e-10
        
    def get_spectrEk(self,kran,**kwargs):
        '''Extract spectrum as function of momentum E(k)'''
        spectr=zeros((len(kran),self.dim))
        for i in range(len(kran)):
            k=kran[i]
            spectr[i,:]=eigvalsh(self.H0+exp(1.0j*k)*self.H1+exp(-1.0j*k)*self.H1.H)
            
        return spectr    
    
    def set_ene(self,E,**kwargs):
        '''
        Build energy dependent quantities
        
        This is the part where the bulk and surface 
        Green's function and self-energies are calculated.
        
        '''
        
        self.E=E                             # This is the enregy we are using
        Z=zeros_like(self.H0)                # Zero matrix and Identity matrix
        Id=eye(self.dim)                     # of size dimXdim
        
        # Setting up the tricky eigenvalue problem ...
        A =vstack((hstack((E*eye(self.dim)-self.H0,-self.H1.H)),
                   hstack((Id,Z))))        
        B =vstack((hstack((self.H1,Z)),
                   hstack((Z,Id))))
        # and solving it.
        w,vec=sl.eig(A,B)
        self.w=w
        
        # get k if asked for
        if kwargs.get('get_k'):
            self.k=-1.0j*log(w) 

        # Obtaining and normailizing the eigen vectors of the transverse modes
        vec=matrix(vec[:self.dim,:])
        vg=(zeros_like(w));
        for i in range(2*self.dim):
            vec[:,i]=vec[:,i]/norm(vec[:,i])              
            vg[i]=1.0j*((vec[:,i].H*(w[i]*self.H1-w[i]**(-1)*self.H1.H)*vec[:,i])[0,0])
        self.vg=vg
        
        # Sorting left and right channels in to propagaintg and decaying
        
        opened=where((abs(abs(w)-1)<self.tol))
        closed=where(abs(abs(w)-1)>self.tol)

        left =union1d(intersect1d(opened,where(real(vg)<0)),     
                      intersect1d(closed,where(abs(w)>1)))
        right=union1d(intersect1d(opened,where(real(vg)>0)),     
                      intersect1d(closed,where(abs(w)<1)))
                     
        
        self.vec_left=vec[:,left]
        self.w_left  =w[left]
        self.vg_left =vg[left]
        
        self.vec_right=vec[:,right]
        self.w_right  =w[right]
        self.vg_right =vg[right]
                                            
        if len(self.w_right)!=len(self.w_left):
            print('Problem with partitioning!!')
            return 

        # Obtaining the duals
        self.vec_right_dual=inv(self.vec_right)
        self.vec_left_dual=inv(self.vec_left)    
                      
        # Packaging open channel quantities                                 
        left_open  =where(abs(abs(w[left])-1 )<self.tol)
        right_open =where(abs(abs(w[right])-1 )<self.tol)             
                      
        self.vg_left_open=(self.vg[left])[left_open]
        self.vg_right_open=(self.vg[right])[right_open]

        self.w_left_open=(self.w[left])[left_open]
        self.w_right_open=(self.w[right])[right_open]
                                                                  
        self.vec_left_open =(self.vec_left[:,left_open])[:,0,:]     # for some wierd reson the [...] is needed 
        self.vec_right_open =(self.vec_right[:,right_open])[:,0,:]  # if i want to have a nice matrix
                      
        self.vec_left_dual_open =(self.vec_left_dual[left_open,:])[0,:,:] # same here with the [...]
        self.vec_right_dual_open=(self.vec_right_dual[right_open,:])[0,:,:]
        
        # Get transfere matrices    
        self.T_leftm1  = self.vec_left  * matrix(diag(w[left]))    * self.vec_left_dual
        self.T_left    = self.vec_left  * matrix(diag(1/w[left]))  * self.vec_left_dual
        self.T_right   = self.vec_right * matrix(diag(w[right]))   * self.vec_right_dual
        self.T_rightm1 = self.vec_right * matrix(diag(1/w[right])) * self.vec_right_dual
        
        self.V=self.H1*(self.T_leftm1-self.T_right)
        
        if kwargs.get('get_selfE'):
        # Get self-energy
            self.selfEL=self.H1.H*self.T_left
            self.selfER=self.H1*self.T_right
        elif kwargs.get('get_bulkG'):
        # Get Green's functions of infinite bulk and leads
            self.g00=inv(self.V)
            self.gsL=self.T_left*inv(self.H1)
            self.gsR=self.T_right*inv(self.H1.H)

        else:
            self.gsL=self.T_left*inv(self.H1)
            self.gsR=self.T_right*inv(self.H1.H)
    
    def Tz(self,z,**kwargs):
        ''' 
        A function to obtain arbitrary tranfere matrices
        '''
        if (hasattr(self,'E')):
            return (self.vec_left  * matrix(diag(self.w_left**z))    * self.vec_left_dual,
                    self.vec_right  * matrix(diag(self.w_right**z))    * self.vec_right_dual)
        else:
            print('Set energy first')