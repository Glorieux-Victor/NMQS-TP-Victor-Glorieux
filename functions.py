import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt
from itertools import combinations

def hermitian_with_spectrum(eigenvalues, seed=0, dtype=np.complex128):
    """Dense complex Hermitian matrix with *exactly* the requested spectrum.
    H = Q diag(lambda) Q^dagger with Q Haar-distributed unitary.  This is the
    workhorse for controlled experiments: the reference answer is known
    analytically, so measured errors are not contaminated by the reference.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    D = lam.size
    rng = np.random.default_rng(seed)
    Z = (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    Q = Q * (np.diag(R) / np.abs(np.diag(R)))          # fix the QR phase -> Haar measure
    return ((Q * lam) @ Q.conj().T).astype(dtype)


def hermiticity_error(H):
    """||H - H^dagger||_F / ||H||_F -- always check this before diagonalising."""
    if sp.issparse(H):
        D = H - H.conj().T
        return spla.norm(D) / spla.norm(H)
    return np.linalg.norm(H - H.conj().T) / np.linalg.norm(H)


def _matvec_and_dim(H, dtype):
    """
        Return (matvec, D), casting H to `dtype` if it is not already of dtype.

        matvec is a function, with
        matvec(v) = H @ v for three different data types used here.

        D is the dimension of H.
    """
    if sp.issparse(H) or isinstance(H, np.ndarray):
        Hc = H if H.dtype == dtype else H.astype(dtype)
        return (lambda x: Hc @ x), H.shape[0]
    if isinstance(H, spla.LinearOperator):
        return (lambda x: H.matvec(x)), H.shape[0]
    raise TypeError("H must be a numpy array, a scipy sparse matrix, or a LinearOperator")

def _randomvec( D, dtype=np.complex128, seed=0 ):
    '''
    Creates a D-element normalized random vector of type dtype. Will come in handy when implementing
    Lanczos.
    '''
    rng = np.random.default_rng(seed)
    phi0 = rng.normal(size=D) + 1j * rng.normal(size=D)
    phi0 = np.asarray(phi0, dtype)
    phi0 = phi0 / np.linalg.norm(phi0)
    return phi0

def _project_out( Phid, n, gs_passes=2 ):
    '''
    Gram-Schmidt procedure, done gs_passes times
    '''
    for _ in range(gs_passes): # do this at least twice
        h = Phid[:,:n+1].conj().T @ Phid[:,n+1]
        # size check nxD times Dx1 is a nx1 vector of weigths,
        # i.e. inner products of the current vector with
        # the n basis vectors prior to it. Subtract the components
        # of this vector onto the vectors prior to it.
        Phid[:,n+1] -= Phid[:,:n+1] @ h
        # size check: Dx1 on lhs and Dxn . nx1 = Dx1 on rhs

# lanczos_matrix
def lanczos_matrix( H, d, phi0=None, dtype=np.complex128, gs_passes=2, return_evecs=True ):
    '''
    Subroutine of the lanczos algorithm: Returns the elements of the Lanczos tridiagonal matrix,
    as alpha, and beta, along with the Krylov vectors. Note that the tridiagonla matrix should be
    Hd whose diagonal is $alpha_0, ... ,alpha_{d-1}$, with d the size of the Krylov subspace, while
    diagonal +/- 1 contains $ beta_{1}, ... , beta_{d-1}$.
    '''
    matvec, D  = _matvec_and_dim(H, dtype)

    alpha = np.zeros( d+1, dtype = dtype )
    beta  = np.zeros( d+1, dtype = dtype )
    Phid  = np.zeros( (D,d+1), dtype = dtype  )
    Phid[:,0] = phi0  # the first Krylov vector

    for n in range(0, d):
        phiprime = matvec(Phid[:,n])
        if (n>0):
            phiprime -= beta[n] * Phid[:,n-1]
        alpha[n] = np.vdot(Phid[:,n], phiprime)
        phisecond = phiprime - alpha[n] * Phid[:,n]
        beta[n+1] = np.linalg.norm(phisecond)
        Phid[:,n+1] = phisecond/beta[n+1]

        # if gs_passes is zero
        _project_out( Phid, n, gs_passes=gs_passes )

    return {'alpha': alpha, # note one extra value of alpha, alpha[d] is returned
            'beta' : beta,  # analogous, beta[d] is also returned
            'Phid' : Phid if return_evecs else None }


def lanczos( H, d=10, phi0=None, k=0, dtype=np.complex128, seed=0, gs_passes=2 ):
    '''
    H: linear operator to diagonalize,
    d: dimension of the Krylov subspace,
    phi0: initial ket,
    k: number of eigenvectors to return
    dtype: data type for the entries of all arrays, defaults to numpy complex128
    seed: random number generator seed, should be an integer
    gs_passes: number of Gram-Schmidt passes. Set to 0 if you implement no reorthogonalization
    '''
    if phi0 == None:
        phi0 = _randomvec( np.shape(H)[0], dtype=dtype, seed=seed )
    lm = lanczos_matrix( H, d, phi0, dtype=dtype, gs_passes=gs_passes, return_evecs=True )

    # Ritz values theta[i] and Ritz vectors psi[i], for i = 0, ..., d-1
    theta, psi = ritz(lm, d)
    # psi[m,j] contains the the nth entry of the mth Ritz vector, and psi is a dxd matrix

    return {
            'theta': theta, # Ritz values
            'psi'  : psi,   # Ritz vectors
            'Psi'  : (lm['Phid'][:,:-1] @ psi)[:,:k] if k>0 else None,
                            # a matrix containing the approximations to the first k eigenvectors, as obtained
                            # from the Ritz vectors
            'alpha': lm['alpha'],
            'beta':  lm['beta']
           }

def ritz(lanczos_output, d=None):
    '''
    This returns the Ritz values and vectors from the output of the lanczos function,
    at a given Krylov subspace dimension d. Useful for convergence tests.
    '''
    if d==None:
        d=len(lanczos_output['alpha'])
    theta, psi = eigh_tridiagonal( np.real(lanczos_output['alpha'][0:d]),
                                   np.real(lanczos_output['beta'][1:d]) )
    return  theta, psi    # Ritz values and vectors



def basis_states(L, N):
    """Occupation-number basis of the N-particle sector, as sorted bitstrings."""
    states = np.array(sorted(sum(1 << j for j in sites)
                             for sites in combinations(range(L), N)), dtype=np.int64)
    return states, {int(s): i for i, s in enumerate(states)}


def occ(j, s):
    """Is site j occupied in the bitstring state s?

    Example: s = 13, bin(s) = '0b1101', so occ(0, s) = 1 but occ(1, s) = 0.
    """
    return (s >> j) & 1

# s is an integer, bin(s) just gives the binary representation of s.


def c(j, s):
    """Remove a particle at site j: if the jth bit is 1 it is set to 0, else nothing.

    For that `else' the caller needs an external guardrail -- see h_fermion_ring.
    """
    return s & ~(1 << j)


def cdag(j, s):
    """Add a particle at site j: if the jth bit is 0 it is set to 1, else nothing.

    For that `else' the caller needs an external guardrail -- see h_fermion_ring.
    """
    return s | (1 << j)


def count_below(j, s):
    """Number of particles on sites with index strictly below j (Jordan-Wigner string).

    (1 << j) is 2^j; (1 << j) - 1 is 2^j - 1 = 2^0 + ... + 2^(j-1), i.e. a mask of
    ones on every site below j.  For j = 3 that is 1000 - 1 = 111.  The & then keeps
    only those sites of s, and .count("1") counts the particles among them.
    """
    return bin(s & ((1 << j) - 1)).count("1")


def free_fermion_E0(L, N, t=1.0):
    """Exact ground-state energy of the non-interacting on a flux-threaded ring: fill the N lowest levels."""
    eps = -2.0 * t * np.cos((2 * np.pi * np.arange(L)) / L)
    return np.sort(eps)[:N].sum()

    
#==================================================================================================================================
#==================================================================================================================================

def h_fermion_ring(L, N, t=1.0, dtype=np.complex128):
    """Hopping matrix for spinless fermions on an L-site ring threaded by flux, N-particle sector (CSR scipy matrix).
    Pure kinetic energy: -t sum_j c^dag_{j+1} c_j + h.c.), with
    Jordan-Wigner signs.
    """
    states, indices = basis_states(L, N)
    dim = len(states)
    rows, cols, vals = [], [], []
    H =  np.zeros((dim, dim))


    for s in states:
        a=indices[s]
        for j in range(L): #j+1 mod L => keep periodicity
            k = (j+1)%L
            for (src, dst, amp) in [ (j, k, -t),        # forward hop, src = source site, dst : destination site
                                     (k, j, -t) ]:      # its Hermitian partner

                if occ(src, s)==0 or occ(dst, s)==1:
                    continue         # c|0> = 0, or cdag|1> = 0

                sign  = (-1)**(count_below(src, s))   # string of c_src
                s1    = c(src,s)                      # the INTERMEDIATE state
                sign *= (-1)**(count_below(dst, s1))   # string of cdag_dst
                b     = indices[cdag(dst,s1)] #find the entry

                #H[b, a] += amp * sign #fill the H matrix

                rows.append(b)
                cols.append(a)
                vals.append(amp * sign)

                
    #
    #
    # Assign matrix elements (stored in sparse format as shown in the line right below)
    # according to the pseudocode above.
    #
    #
    #H_= H #to see the matrix
    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()
    H.sum_duplicates()
    
    #return H.astype(dtype), states,H_
    return H.astype(dtype), states




def h_fermion_ring_open(L, N, t=1.0, dtype=np.complex128):
    """Hopping matrix for spinless fermions on an L-site ring threaded by flux, N-particle sector (CSR scipy matrix).
    Pure kinetic energy: -t sum_j c^dag_{j+1} c_j + h.c.), with
    Jordan-Wigner signs.
    """
    states, indices = basis_states(L, N)
    dim = len(states)
    rows, cols, vals = [], [], []
    H =  np.zeros((dim, dim))


    for s in states:
        a=indices[s]
        for j in range(L-1): #j+1 mod L => keep periodicity
            k = (j+1)
            for (src, dst, amp) in [ (j, k, -t),        # forward hop, src = source site, dst : destination site
                                     (k, j, -t) ]:      # its Hermitian partner

                if occ(src, s)==0 or occ(dst, s)==1:
                    continue         # c|0> = 0, or cdag|1> = 0

                sign  = (-1)**(count_below(src, s))   # string of c_src
                s1    = c(src,s)                      # the INTERMEDIATE state
                sign *= (-1)**(count_below(dst, s1))   # string of cdag_dst
                b     = indices[cdag(dst,s1)] #find the entry

                #H[b, a] += amp * sign #fill the H matrix

                rows.append(b)
                cols.append(a)
                vals.append(amp * sign)

                
    #
    #
    # Assign matrix elements (stored in sparse format as shown in the line right below)
    # according to the pseudocode above.
    #
    #
    #H_= H #to see the matrix
    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()
    H.sum_duplicates()
    
    #return H.astype(dtype), states,H_
    return H.astype(dtype), states



#==================================================================================================================================
#==================================================================================================================================



def dimer_block_11(U, t):
    """The (N_up, N_dn) = (1,1) block, written out by hand in the |A>,|B>,|C>,|D> basis."""
    return np.array([[U, -t, -t,  0],
                     [-t,  0,  0, -t],
                     [-t,  0,  0, -t],
                     [0, -t, -t,  U]], dtype=float)


def dimer_spectrum(U, t):
    """All 16 eigenvalues of the two-site Hubbard model, in closed form."""
    R = np.sqrt(U**2 + 16 * t**2)
    return np.sort([0.0,                               # N=0
                    -t, +t, -t, +t,                    # N=1
                    0.0, 0.0, 0.0,                     # N=2, S=1 triplet
                    U, (U - R) / 2, (U + R) / 2,       # N=2, S=0 singlets
                    U - t, U - t, U + t, U + t,        # N=3
                    2 * U])                            # N=4


def hubbard_operator(L, Nup, Ndn,model='periodic', t=0.5, U=10.):
    """
        Matrix-free Fermi-Hubbard H in the (Nup, Ndn) sector.
        Returns (LinearOperator, double-occupancy diagonal, (D_up, D_dn)).
    """

    if model == 'periodic' :
    
        # Hamiltonian for up :
        Hup = h_fermion_ring(L, Nup, t, dtype=np.complex128)
        # Hamiltonian for dn :
        Hdn = h_fermion_ring(L, Ndn, t, dtype=np.complex128)

    if model == 'open' : 

        # Hamiltonian for up :
        Hup = h_fermion_ring_open(L, Nup, t, dtype=np.complex128)
        # Hamiltonian for dn :
        Hdn = h_fermion_ring_open(L, Ndn, t, dtype=np.complex128)
    
    su_init, indices_up = basis_states(L, Nup)
    Du = len(su_init)

    sd_init, indices_dn = basis_states(L, Ndn)
    Dd = len(sd_init)


    # Dataframe to see where the states are with the indexes I :
    I, Iu, Id, states_u, states_d = [], [], [], [], []

    for i in range(Du):
        for j in range(Dd):
            I.append(i*Du+j)
            Iu.append(i)
            Id.append(j)
            states_u.append(int(su_init[i]))
            states_u_bin = np.array([bin(i) for i in states_u])
            states_d.append(int(su_init[j]))
            states_d_bin = np.array([bin(i) for i in states_d])


    hub_states_data = {'I' : I,'Iu' : Iu, 'state_u' : states_u_bin,'Id' : Id, 'state_d' : states_d_bin}
    #hub_states = pd.DataFrame(data=hub_states_data)

    
    docc = np.array([[bin(int(a) & int(b)).count("1") for b in sd_init]
                 for a in su_init], dtype=np.float64).ravel()
        

    def matvec(x):
        P = x.reshape(Du, Dd)
        #print('\nHup[0] @ P :\n',Hup[0] @ P)
        #print('\n(Hdn[0] @ P.T).T :\n',(Hdn[0] @ P.T).T)
        #print('\n(Hup[0] @ P + (Hdn[0] @ P.T).T).ravel() :\n',(Hup[0] @ P + (Hdn[0] @ P.T).T).ravel())
        #print('\n(U * docc) : \n',(U * docc))
        #print(x)
        return (Hup[0] @ P + (Hdn[0] @ P.T).T).ravel() + (U * docc) * x


    return spla.LinearOperator((Du * Dd, Du * Dd), matvec=matvec,
                                dtype=np.complex128), docc, (Du, Dd)


def hubbard_dense(L, Nup, Ndn, **kw):
    """The same H, assembled explicitly.  Only for validation at small L."""
    op, docc, (Du, Dd) = hubbard_operator(L, Nup, Ndn, **kw)
    D = Du * Dd
    return np.column_stack([op.matvec(e) for e in np.eye(D, dtype=np.complex128)])