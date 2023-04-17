import numpy as np



def crm(p,iters=3):
    """continous response model. Expects np.array of shape (students, questions)"""
    if np.any(p>1) or np.any(p<0):
        raise ValueError("p must be in [0,1]")

    N, M = p.shape

    def goz(p):
        p=np.maximum(p,0.0001)
        p=np.minimum(p,0.9999)
        return np.log(p/(1-p))
    def _goz(p):
        p=0.0001+0.9998*p
        return np.log(p/(1-p))

    z=goz(p)

    def theta_from_var(alpha,beta,gamma):
        thetas=[]
        for i in range(N):
            theta=0.0
            divide=0.0
            for j in range(M):
                theta+=(alpha[j]**2)*(beta[j]+gamma[j]*z[i,j])
                divide+=alpha[j]**2
            theta/=divide
            thetas.append(theta)
        thetas=np.array(thetas)
        return thetas

    def var_from_theta(theta):
        M=np.mean(theta)
        Mj=np.mean(z,axis=0)
        Cj=np.mean(theta*z.T,axis=1)-M*Mj#I think p.T is correct
        #VT=np.var(theta)#write this out?
        VT=np.mean(theta**2)-M**2
        gamma=VT/Cj
        beta=M-gamma*Mj
        #Vz=np.var(z,axis=0)#this too?
        Vz=np.mean(z**2,axis=0)-Mj**2
        Vj=np.abs(VT-2*gamma*Cj+(gamma**2)*Vz)
        alpha=1/(np.sqrt(Vj)+1e-6)
        #print(M.shape,Mj.shape,Cj.shape,VT.shape,gamma.shape,beta.shape,Vz.shape,Vj.shape,alpha.shape)
        #exit()
        return alpha,beta,gamma



    alpha,beta,gamma=var_from_theta(np.mean(z,axis=1))

    thetas=theta_from_var(alpha,beta,gamma)
    for i in range(iters-1):
        alpha,beta,gamma=var_from_theta(thetas)
        print(np.sum(alpha),np.sum(beta),np.sum(gamma))
        thetas=theta_from_var(alpha,beta,gamma)
    return thetas

#def crm(p):
#    return np.mean(p,axis=1)

def crm(p, delta_border=0.01, iter_max=100):
    """continous response model. Expects np.array of shape (students, questions)"""
    if np.any(p>1) or np.any(p<0):
        print(np.min(p,axis=0),np.max(p,axis=0))
        raise ValueError("p must be in [0,1]")

    N, M = p.shape

    def goz(p):
        p = np.maximum(p, 0.0001)
        p = np.minimum(p, 0.9999)
        return np.log(p / (1 - p))

    def goz(p):
        p = 0.0001 + 0.9998 * p
        return np.log(p / (1 - p))

    z = goz(p)

    def estEM(data,alpha,beta,gamma):
        sigma=1/(np.sum(alpha**2)+1)
        #        mu <-sigma*rowSums(t(matrix(ipar[,1]^2,ncol=N,nrow=n))*(t(matrix(ipar[,3],ncol=N,nrow=n))*data+t(matrix(ipar[,2],ncol=N,nrow=n))),na.rm=TRUE) # Equation 20 in Shojima's paper
        #print(sigma.shape,alpha.shape,gamma.shape,(gamma*z).shape,z.shape,beta.shape)
        #print((gamma*z+beta).shape)
        #print((alpha**2*(gamma*z+beta)).shape)
        #exit()
        mu=sigma*np.sum(alpha**2*(gamma*z+beta),axis=0)#entirely copilot
        mumean=np.mean(mu)
        muvar=np.var(mu)
        zijmeanlist=np.mean(z,axis=0)#this and next 2 mostly copilot
        zijvarlist=np.var(z,axis=0)
        zijmucovlist=np.mean(z*mu,axis=0)-mumean*zijmeanlist

        gamma=(muvar+sigma)/zijmucovlist
        beta=mumean-gamma*zijmeanlist
        alpha=1/(np.sqrt((gamma**2)*zijvarlist+gamma*zijmucovlist))
        return alpha,beta,gamma

    def theta_from_var(alpha,beta,gamma):
        thetas=[]
        for i in range(N):
            theta=0.0
            divide=0.0
            for j in range(M):
                theta+=(alpha[j]**2)*(beta[j]+gamma[j]*z[i,j])
                divide+=alpha[j]**2
            theta/=divide
            thetas.append(theta)
        thetas=np.array(thetas)
        return thetas




    alpha,beta,gamma=estEM(z,np.ones(M),np.zeros(M),np.ones(M)-np.mean(z,axis=0))
    last_alpha,last_beta,last_gamma=alpha,beta,gamma
    thetas=theta_from_var(alpha,beta,gamma)
    for i in range(iter_max):
        alpha,beta,gamma=estEM(z,alpha,beta,gamma)
        delta=np.sum(np.abs(alpha-last_alpha))+np.sum(np.abs(beta-last_beta))+np.sum(np.abs(gamma-last_gamma))
        if delta<delta_border:
            break
        last_alpha,last_beta,last_gamma=alpha,beta,gamma

    thetas=theta_from_var(alpha,beta,gamma)
    return thetas, (alpha,beta,gamma)

def crm(p, delta_border=0.01, iter_max=10):
    """continous response model. Expects np.array of shape (students, questions)"""
    #modification with many abs etc
    #print("hello")
    if np.any(p>1) or np.any(p<0):
        print(np.min(p,axis=0),np.max(p,axis=0))
        raise ValueError("p must be in [0,1]")

    N, M = p.shape

    def goz(p):
        p = np.maximum(p, 0.0001)
        p = np.minimum(p, 0.9999)
        return np.log(p / (1 - p))

    def goz(p):
        p = 0.0001 + 0.9998 * p
        return np.log(p / (1 - p))

    z = goz(p)

    def estEM(data,alpha,beta,gamma):
        
        sigma=1/(np.sum(alpha**2)+1)
        #        mu <-sigma*rowSums(t(matrix(ipar[,1]^2,ncol=N,nrow=n))*(t(matrix(ipar[,3],ncol=N,nrow=n))*data+t(matrix(ipar[,2],ncol=N,nrow=n))),na.rm=TRUE) # Equation 20 in Shojima's paper
        #print(sigma.shape,alpha.shape,gamma.shape,(gamma*z).shape,z.shape,beta.shape)
        #print((gamma*z+beta).shape)
        #print((alpha**2*(gamma*z+beta)).shape)
        #exit()
        mu=sigma*np.sum(alpha**2*(gamma*z+beta),axis=0)#entirely copilot
        mumean=np.mean(mu)
        muvar=np.var(mu)
        zijmeanlist=np.mean(z,axis=0)#this and next 2 mostly copilot
        zijvarlist=np.var(z,axis=0)
        zijmucovlist=np.mean(z*mu,axis=0)-mumean*zijmeanlist

        gamma=(muvar+sigma)/zijmucovlist
        beta=mumean-gamma*zijmeanlist
        alpha=1/(np.sqrt(np.abs((gamma**2)*zijvarlist+gamma*zijmucovlist)+1e-10))
        return alpha,beta,gamma

    def theta_from_var(alpha,beta,gamma):
        if np.any(np.isnan(alpha)) or np.any(np.isnan(beta)) or np.any(np.isnan(gamma)):
            return np.mean(data,axis=-1)
        thetas=[]
        for i in range(N):
            theta=0.0
            divide=0.0
            for j in range(M):
                theta+=(alpha[j]**2)*(beta[j]+gamma[j]*z[i,j])
                divide+=alpha[j]**2
            theta/=divide+1e-10
            thetas.append(theta)
        thetas=np.array(thetas)
        return thetas




    alpha,beta,gamma=estEM(z,np.ones(M),np.zeros(M),np.ones(M)-np.mean(z,axis=0))
    last_alpha,last_beta,last_gamma=alpha,beta,gamma
    thetas=theta_from_var(alpha,beta,gamma)
    #print(thetas)
    for i in range(iter_max):
        alpha,beta,gamma=estEM(z,alpha,beta,gamma)
        if np.any(np.isnan(alpha)) or np.any(np.isnan(beta)) or np.any(np.isnan(gamma)):
            alpha,beta,gamma=last_alpha,last_beta,last_gamma
            break
        #print(np.any(np.isnan(alpha)),np.any(np.isnan(beta)),np.any(np.isnan(gamma)))
        delta=np.sum(np.abs(alpha-last_alpha))+np.sum(np.abs(beta-last_beta))+np.sum(np.abs(gamma-last_gamma))
        if delta<delta_border:
            break
        last_alpha,last_beta,last_gamma=alpha,beta,gamma

    alpha=np.abs(alpha)
    beta=np.abs(beta)
    gamma=np.abs(gamma)

    thetas=theta_from_var(alpha,beta,gamma)
    return thetas, (alpha,beta,gamma)


if __name__=="__main__":
    student1=[1.0,1.0,1.0,0.0,0.0]
    #student1=[1.0,1.0,0.0,0.0,0.0]
    student2=[1.0,0.0,0.0,1.0,0.0]
    student3=[1.0,1.0,0.0,0.0,0.0]
    students=np.stack([student1,student2,student3])
    theta=crm(students)
    print(theta)






