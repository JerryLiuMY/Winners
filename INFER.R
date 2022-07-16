# INFERENCE ON WINNERS
# AUTHORS:      I.ANDREWS (HARVARD UNIVERSITY)
#               T.KITAGAWA (UNIVERSITY COLLEGE LONDON)
#               A.MCCLOSKEY (UNIVERSITY OF COLORADO)
# CODE AUTHOR:  J.ROWLEY (UNIVERSITY COLLEGE LONDON; CONTRIBUTIONS BY AUTHORS)
# THIS VERSION: JUNE 7TH 2021

## -------------------------------------------------------------------------- ##

# DESCRIPTION:  CODE DESIGNATED FOR GENERAL USE WITH GENERAL ESTIMATES AND 
#               ESTIMATED VARIANCE MATRIX.
#               CODE PERFORMS INFERENCE ON WINNERS.
#               CODE WRITTEN AS SCRIPT TO PRESERVE THE SEED INSIDE THE ROOT-
#               FINDING ALGORITHMS THAT ARE USED, WHICH SEARCH ROUTINES INSIDE
#               FUNCTIONS DO NOT EASILY PRESERVE.

## -------------------------------------------------------------------------- ##

# REQUIRED PACKAGES INCLUDE:  <<matrixcalc>> 
#                             <<MatrixStats>> 
#                             <<TruncatedNormal>> 

## -------------------------------------------------------------------------- ##

# USER INPUTS:  X         VECTOR OF ESTIMATES OF RETURN TO EACH TREATMENT ARM.
#               SIGMA     VARIANCE MATRIX OF ESTIMATES.
#               NDRAWS    NUMBER OF BOOTSTRAP DRAWS TO COMPUTE TAIL MASS.
#               ALPHA     SIGNIFICANCE LEVEL. DEFAULT 0.05.
#               BETA      SIGNIFICANCE LEVEL. DEFAULT 0.005.

X <- c(0.813267813267813,0.612068965517241,0.713670613562971,0.922330097087379,0.81270182992465,
       0.980603448275862,0.891891891891892,0.780409041980624,1.51619870410367,0.786637931034482,
       0.92572658772874,1.27076591154261,1.02909482758621,1.4217907227616,0.81270182992465,
       1.50699677072121,0.575135135135135,1.33620689655172,1.02047413793103,0.528586839266451,
       0.862068965517241,1.11530172413793,1.10118406889128,1.22306034482759,0.808189655172414,
       1.24379719525351,0.905172413793104,1.40280777537797,0.811218985976268,0.88133764832794,
       0.69073275862069,1.26049515608181,0.579288025889968,0.702586206896552,0.890301724960064,
       0.964439655172413,0.92170626349892);
SIGMA <- diag(c(0.00400616351128161,0.030059017384886,0.102029725567161,0.088658020114521,0.104295636149854,
                0.0737445782342327,0.0654365980297311,0.0518042635802516,0.12079318490481,0.0684821760626574,
                0.0644857835659649,0.255533140108395,0.102640335212518,0.153208002787522,0.0592803935368147,
                0.211508149115732,0.0401274251080888,0.184853763479433,0.0659587393534699,0.0212852032792557,
                0.0517431174146126,0.0817138529228554,0.0995694163979758,0.0917188736769625,0.071057820115819,
                0.12137947379161,0.0620537018481698,0.130066325704489,0.0460172568626761,0.0563916115263375,
                0.0371677136381132,0.0955191702042282,0.0330349869841438,0.0397080226677499,0.090992021208343,
                0.0892513406062135,0.0530575476212782));
NDRAWS <- 10000;
ALPHA <- 0.05;
BETA <- 0.005;

FEED.IN <- matrix(rnorm(length(X)*NDRAWS),nrow=NDRAWS);
# DRAW FROM THE STANDARD NORMAL DISTRIBUTION WITH FIXED SEED.

Y <- X;
# REPLICATE THE VECTOR OF ESTIMATES.
SIGMA <- kronecker(matrix(rep(1,4),ncol=2),SIGMA);
# REPLICATE THE VARIANCE MATRIX.

## -------------------------------------------------------------------------- ##
## PRESERVED FUNCTIONS
## -------------------------------------------------------------------------- ##

ETRN2 <- function(MU,A,B,SIGMA,N,SEED=100){
  set.seed(SEED);
  TAIL.PROB <- SIGMA*mean(norminvp(p=runif(N),
                                   l=rep((A-MU)/SIGMA,N),
                                   u=rep((B-MU)/SIGMA,N)))+MU;
  return(TAIL.PROB)
}
# COMPUTATION OF THE MEAN OF THE TRUNCATED NORMAL DISTRIBUTION, WHERE (A,B) ARE
# THE TRUNCATION POINTS AND THE MEAN IS MU.

PTRN2 <- function(MU,Q,A,B,SIGMA,N,SEED=100){
  set.seed(SEED);
  TAIL.PROB <- mean((norminvp(p=runif(N),
                              l=rep((A-MU)/SIGMA,N),
                              u=rep((B-MU)/SIGMA,N)) <= ((Q-MU)/SIGMA)));
  return(TAIL.PROB)
}
# APPROXIMATION OF THE CUMULATIVE DISTRIBUTION FUNCTION (I.E., X[I] <= Q) OF THE
# TRUNCATED NORMAL DISTRIBUTION, WHERE WHERE (A,B) ARE THE TRUNCATION POINTS AND 
# THE MEAN IS MU.

# NB: THE FUNCTION SOLVING EQ.14 FOR MU, <<FTN(Y-HAT;MU,A,B) >  > , IS THE TRUNCATED
#     NORMAL OF XI ~ N(MU,SIGMA-Y-HAT) TRUNCATED AT (A,B).
# NB: N IS THE NUMBER OF DRAWS TO APPROXIMATE THE TRUNCATED NORMAL CUMULATIVE 
#     DISTRIBUTION FUNCTIONS.

## -------------------------------------------------------------------------- ##
## ADDITIONAL FUNCTIONS
## -------------------------------------------------------------------------- ##

CUTRN <- function(MU,Q,A,B,SIGMA,SEED=100){
  set.seed(SEED);
  CUT <- SIGMA*norminvp(p=Q,l=(A-MU)/SIGMA,u=(B-MU)/SIGMA)+MU;
  return(CUT)
}
# FINDS THE THRESHOLD FOR CONFIDENCE REGION EVALUATION.

CHYRN <- function(MU,Q,A,B,SIGMA,CV_BETA,SEED=100){
  set.seed(SEED);
  CUT <- SIGMA*norminvp(p=Q,
                        l=max((A-MU)/SIGMA,-CV_BETA),
                        u=min((B-MU)/SIGMA,+CV_BETA))+MU;
  return(CUT)
}
# FINDS THE THRESHOLD FOR CONFIDENCE REGION EVALUATION IN THE HYBRID SETTING.

## -------------------------------------------------------------------------- ##
## PRELIMINARIES
## -------------------------------------------------------------------------- ##

K <- length(X);
THETA_TILDE <- which.max(X);
# THE NUMBER OF TREATMENT ARMS AND THE INDEX OF THE WINNING ARM.

YTILDE <- Y[THETA_TILDE];
# THE ESTIMATE ASSOCIATED WITH THE WINNING ARM.

SIGMAY <- SIGMA[(K+1):(2*K),(K+1):(2*K)];
SIGMAYTILDE <- c(SIGMA[K+THETA_TILDE,K+THETA_TILDE]);
SIGMAXYTILDE_VEC <- c(SIGMA[(K+THETA_TILDE),1:K]);
SIGMAXYTILDE <- c(SIGMA[THETA_TILDE,(K+THETA_TILDE)]);
# (I.) VARIANCE OF ALL OF THE ESTIMATES, (II.) VARIANCE OF THE WINNING ARM,
# (III.) COVARIANCE OF THE WINNING ARM AND OTHER ARMS, (IV.) VARIANCE OF THE 
# WINNING ARM.

ZTILDE <- X-(SIGMA[(K+THETA_TILDE),1:K])/SIGMAYTILDE*YTILDE;
# NORMALISED DIFFERENCE BETWEEN EACH TREATMENT ARM AND WINNING ARM.

IND_L <- (SIGMAXYTILDE_VEC<SIGMAXYTILDE);
if(sum(IND_L) ==  0){
  LTILDE <- -Inf;
};
if(sum(IND_L) > 0){
  LTILDE <- max(SIGMAYTILDE*(ZTILDE[IND_L]-ZTILDE[THETA_TILDE])/
                  (SIGMAXYTILDE-SIGMAXYTILDE_VEC[IND_L]));
};
# THE LOWER TRUNCATION VALUE.

IND_U <- (SIGMAXYTILDE_VEC > SIGMAXYTILDE);
if(sum(IND_U) ==  0){
  UTILDE <- +Inf;
};
if(sum(IND_U) > 0){
  UTILDE <- min(SIGMAYTILDE*(ZTILDE[IND_U]-ZTILDE[THETA_TILDE])/
                  (SIGMAXYTILDE-SIGMAXYTILDE_VEC[IND_U]));
};
# THE UPPER TRUNCATION VALUE.

IND_V <- (SIGMAXYTILDE_VEC == SIGMAXYTILDE);
if(sum(IND_V) == 0){
  VTILDE <- 0;
};
if(sum(IND_V) > 0){
  VTILDE <- min(-(ZTILDE[IND_V]-ZTILDE[THETA_TILDE]));
};

GAMMA <- (ALPHA-BETA)/(1-BETA);
# HYBRID LEVEL.

if(is.diagonal.matrix(SIGMAY) == TRUE){
  M.FACTORS <- sqrt(SIGMAY);
};
if(is.diagonal.matrix(SIGMAY) == FALSE){
  M.EIGEN <- eigen(SIGMAY);
  M.FACTORS <- M.EIGEN$vectors%*%
    diag(sqrt(M.EIGEN$values))%*%
    qr.solve(M.EIGEN$vectors);
  rm(M.EIGEN);
};
# COMPUTE THE STANDARD DEVIATIONS OF THE ESTIMATES.

GAUSSIAN_MAX <- colMaxs(t(abs(FEED.IN%*%M.FACTORS))/sqrt(diag(SIGMAY)));
CV_UNIF_BETA <- quantile(GAUSSIAN_MAX,probs=1-BETA);
names(CV_UNIF_BETA) <- NULL;
CV_UNIF <- quantile(GAUSSIAN_MAX,probs=1-ALPHA);
names(CV_UNIF) <- NULL;
# RESCALE THE STANDARD NORMAL DRAWS USING THE STANDARD DEVIATIONS AND COMPUTE
# THE CRITICAL VALUES.

## -------------------------------------------------------------------------- ##
## MED_U_ESTIMATE (MEDIAN UNBIASED ESTIMATE)
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- 0.5;
NMC <- NDRAWS;
# INPUT FUNCTION ARGUMENTS.

CHECK.UNIROOT <- FALSE;
k <- K;
# INITIALISE LOOP.

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  INTERMEDIATE <- sapply(MUGRIDS,PTRN2,
                         Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};
# THE SEARCH ROUTINE RELIES ON THE FACT THAT THERE IS A SINGLE CROSSING POINT.
# THE FUNCTION GENERATES A GRID. IF THE TWO ENDPOINTS OF THAT GRID ARE ON THE 
# SAME SIDE OF THE X-AXIS THEN THE GRID IS EXPANDED UNTIL THE FUNCTION CROSSES
# ZERO.

HALT.CONDITION <- FALSE;
MUGRIDS <- rep(0,3);
# INITIALISE LOOP.

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  PREVIOUS.LINE <- MUGRIDS;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  INTERMEDIATE <- sapply(MUGRIDS,PTRN2,
                         Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  if(max(abs(MUGRIDS-PREVIOUS.LINE)) == 0){
    HALT.CONDITION <- TRUE;
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  PYHAT <- MUGRIDSM;
};
# SIMPLE BISECTION SEARCH ALGORITHM.

MED_U_ESTIMATE <- PYHAT;
# UNBIASED ESTIMATE.

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,PREVIOUS.LINE,PYHAT);
# RESTORE GLOBAL ENVIRONMENT TO ORIGINAL STATE.

## -------------------------------------------------------------------------- ##

# NB: MOST REMAINING PARTS FOLLOW SIMILAR STEPS. AS SUCH, COMMENTING OF CODE IS
#     LIMITED IN REMAINING PARTS UNLESS SOMETHING SUBSTANTIALLY NEW IS 
#     INTRODUCED.

## -------------------------------------------------------------------------- ##

## -------------------------------------------------------------------------- ##
## EQUI_CI_UPPER 
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- 1-ALPHA/2;
NMC <- NDRAWS;

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  INTERMEDIATE <- sapply(MUGRIDS,PTRN2,
                         Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;
MUGRIDS <- rep(0,3);

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  PREVIOUS.LINE <- MUGRIDS;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  INTERMEDIATE <- sapply(MUGRIDS,PTRN2,
                         Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  if(max(abs(MUGRIDS-PREVIOUS.LINE)) == 0){
    HALT.CONDITION <- TRUE;
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  PYHAT <- MUGRIDSM;
};

EQUICI_UPPER <- PYHAT;

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,PREVIOUS.LINE,PYHAT);

## -------------------------------------------------------------------------- ##
## EQUI_CI_LOWER
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- ALPHA/2;
NMC <- NDRAWS;

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  INTERMEDIATE <- sapply(MUGRIDS,PTRN2,
                         Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;
MUGRIDS <- rep(0,3);

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  PREVIOUS.LINE <- MUGRIDS;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  INTERMEDIATE <- sapply(MUGRIDS,PTRN2,
                         Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  if(max(abs(MUGRIDS-PREVIOUS.LINE)) == 0){
    HALT.CONDITION <- TRUE;
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  PYHAT <- MUGRIDSM;
};

EQUICI_LOWER <- PYHAT;

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,PREVIOUS.LINE,PYHAT);

## -------------------------------------------------------------------------- ##
## CONDCI
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- ALPHA;
NMC <- NDRAWS;

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  PU <- sapply(MUGRIDS,PTRN2,
               Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)+(1-SIZE);
  INTERMEDIATE <- rep(2,2);
  if(PU[1] < 1){
    CU <- CUTRN(MU=MUGRIDSL,Q=PU[1],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[1] <- 
      ETRN2(MU=MUGRIDSL,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(PU[2] < 1){
    CU <- CUTRN(MU=MUGRIDSU,Q=PU[2],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[2] <- 
      ETRN2(MU=MUGRIDSU,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;
MUGRIDS <- rep(0,3);

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  PREVIOUS.LINE <- MUGRIDS;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  PU <- sapply(MUGRIDS,PTRN2,
               Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)+(1-SIZE);
  INTERMEDIATE <- rep(2,3);
  if(PU[1] < 1){
    CU <- CUTRN(MU=MUGRIDSL,Q=PU[1],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[1] <- 
      ETRN2(MU=MUGRIDSL,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(PU[2] < 1){
    CU <- CUTRN(MU=MUGRIDSM,Q=PU[2],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[2] <- 
      ETRN2(MU=MUGRIDSM,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSM,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(PU[3] < 1){
    CU <- CUTRN(MU=MUGRIDSU,Q=PU[3],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[3] <- 
      ETRN2(MU=MUGRIDSU,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(max(abs(MUGRIDS-PREVIOUS.LINE)) == 0){
    HALT.CONDITION <- TRUE;
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  CONDCI_UPPER <- MUGRIDSM;
};

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  PL <- sapply(MUGRIDS,PTRN2,Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  INTERMEDIATE <- rep(-2,2);
  if(PL[1] > 0){
    CL <- CUTRN(MU=MUGRIDSL,Q=PL[1],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[1] <- 
      ETRN2(MU=MUGRIDSL,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(PL[2] > 0){
    CL <- CUTRN(MU=MUGRIDSU,Q=PL[2],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[2] <- 
      ETRN2(MU=MUGRIDSU,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;
MUGRIDS <- rep(0,3);
while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  PREVIOUS.LINE <- MUGRIDS;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  PL <- sapply(MUGRIDS,PTRN2,
               Q=YHAT,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
  INTERMEDIATE <- rep(-2,3);
  if(PL[1] > 0){
    CL <- CUTRN(MU=MUGRIDSL,Q=PL[1],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[1] <- 
      ETRN2(MU=MUGRIDSL,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(PL[2] > 0){
    CL <- CUTRN(MU=MUGRIDSM,Q=PL[2],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[2] <- 
      ETRN2(MU=MUGRIDSM,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSM,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(PL[3] > 0){
    CL <- CUTRN(MU=MUGRIDSU,Q=PL[3],A=L,B=U,SIGMA=sqrt(SIGMAYHAT))
    INTERMEDIATE[3] <- 
      ETRN2(MU=MUGRIDSU,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
      ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
  };
  if(max(abs(MUGRIDS-PREVIOUS.LINE)) == 0){
    HALT.CONDITION <- TRUE;
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  CONDCI_LOWER <- MUGRIDSM;
};

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,PREVIOUS.LINE,PU,CU,PL,CL);

## -------------------------------------------------------------------------- ##
## HYBD_MED_ESTIMATE
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- 0.5;
CV_BETA <- CV_UNIF_BETA;
NMC <- NDRAWS;

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  INTERMEDIATE <- rep(2,2);
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  if(L.BOUND[1] < U.BOUND[1]){
    INTERMEDIATE[1] <- PTRN2(MU=MUGRIDSL,
                             Q=YHAT,
                             A=L.BOUND[1]
                             ,B=U.BOUND[1],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[2] < U.BOUND[2]){
    INTERMEDIATE[2] <- PTRN2(MU=MUGRIDSU,
                             Q=YHAT,A=L.BOUND[2],
                             B=U.BOUND[2],
                             SIGMA=sqrt(SIGMAYHAT)
                             ,N=NMC)-(1-SIZE);
  };
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  INTERMEDIATE <- rep(2,3);
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  if(L.BOUND[1] < U.BOUND[1]){
    INTERMEDIATE[1] <- PTRN2(MU=MUGRIDSL,
                             Q=YHAT,
                             A=L.BOUND[1],
                             B=U.BOUND[1],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[2] < U.BOUND[2]){
    INTERMEDIATE[2] <- PTRN2(MU=MUGRIDSM,
                             Q=YHAT,
                             A=L.BOUND[2],
                             B=U.BOUND[2],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[3] < U.BOUND[3]){
    INTERMEDIATE[3] <- PTRN2(MU=MUGRIDSU,
                             Q=YHAT,
                             A=L.BOUND[3],
                             B=U.BOUND[3],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  PYHAT <- MUGRIDSM;
};

MED_HYBD_ESTIMATE <- PYHAT;

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,PYHAT,CV_BETA,L.BOUND,U.BOUND);

## -------------------------------------------------------------------------- ##
## EQUICI_HYBD_UPPER
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- 1-GAMMA/2;
CV_BETA <- CV_UNIF_BETA;
NMC <- NDRAWS;

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  INTERMEDIATE <- rep(2,2);
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  if(L.BOUND[1] < U.BOUND[1]){
    INTERMEDIATE[1] <- PTRN2(MU=MUGRIDSL,
                             Q=YHAT,
                             A=L.BOUND[1],
                             B=U.BOUND[1],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[2] < U.BOUND[2]){
    INTERMEDIATE[2] <- PTRN2(MU=MUGRIDSU,
                             Q=YHAT,
                             A=L.BOUND[2],
                             B=U.BOUND[2],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  INTERMEDIATE <- rep(2,3);
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  if(L.BOUND[1] < U.BOUND[1]){
    INTERMEDIATE[1] <- PTRN2(MU=MUGRIDSL,
                             Q=YHAT,
                             A=L.BOUND[1],
                             B=U.BOUND[1],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[2] < U.BOUND[2]){
    INTERMEDIATE[2] <- PTRN2(MU=MUGRIDSM,
                             Q=YHAT,
                             A=L.BOUND[2],
                             B=U.BOUND[2],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[3] < U.BOUND[3]){
    INTERMEDIATE[3] <- PTRN2(MU=MUGRIDSU,
                             Q=YHAT,
                             A=L.BOUND[3],
                             B=U.BOUND[3],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  PYHAT <- MUGRIDSM;
};

EQUICI_HYBD_UPPER <- PYHAT;

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,PYHAT,CV_BETA,L.BOUND,U.BOUND);

## -------------------------------------------------------------------------- ##
## EQUICI_HYBD_LOWER
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- GAMMA/2;
CV_BETA <- CV_UNIF_BETA;
NMC <- NDRAWS;

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  INTERMEDIATE <- rep(2,2);
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  if(L.BOUND[1] < U.BOUND[1]){
    INTERMEDIATE[1] <- PTRN2(MU=MUGRIDSL,
                             Q=YHAT,
                             A=L.BOUND[1],
                             B=U.BOUND[1],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[2] < U.BOUND[2]){
    INTERMEDIATE[2] <- PTRN2(MU=MUGRIDSU,
                             Q=YHAT,
                             A=L.BOUND[2],
                             B=U.BOUND[2],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  INTERMEDIATE <- rep(2,3);
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  if(L.BOUND[1] < U.BOUND[1]){
    INTERMEDIATE[1] <- PTRN2(MU=MUGRIDSL,
                             Q=YHAT,
                             A=L.BOUND[1],
                             B=U.BOUND[1],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[2] < U.BOUND[2]){
    INTERMEDIATE[2] <- PTRN2(MU=MUGRIDSM,
                             Q=YHAT,
                             A=L.BOUND[2],
                             B=U.BOUND[2],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if(L.BOUND[3] < U.BOUND[3]){
    INTERMEDIATE[3] <- PTRN2(MU=MUGRIDSU,
                             Q=YHAT,
                             A=L.BOUND[3],
                             B=U.BOUND[3],
                             SIGMA=sqrt(SIGMAYHAT),
                             N=NMC)-(1-SIZE);
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  PYHAT <- MUGRIDSM;
};

EQUICI_HYBD_LOWER <- PYHAT;

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,PYHAT,CV_BETA,L.BOUND,U.BOUND);

## -------------------------------------------------------------------------- ##
## HYBDCI
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
L <- LTILDE;
U <- UTILDE;
SIZE <- ALPHA;
CV_BETA <- CV_UNIF_BETA;
NMC <- NDRAWS;

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  VL <- L.BOUND >= U;
  VU <- U.BOUND <= L;
  INTERMEDIATE <- 2*(VU-VL);
  if(INTERMEDIATE[1] == 0){
    PU <- PTRN2(MU=MUGRIDSL,
                Q=YHAT,
                A=L.BOUND[1],
                B=U.BOUND[1],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)+(1-SIZE);
    if(PU < 1){
      CU <- CHYRN(MU=MUGRIDSL,Q=PU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[1] <- 
        ETRN2(MU=MUGRIDSL,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PU >= 1){
      INTERMEDIATE[1] <- 2;
    };
  };
  if(INTERMEDIATE[2] == 0){
    PU <- PTRN2(MU=MUGRIDSU,Q=YHAT,A=L.BOUND[2],B=U.BOUND[2],SIGMA=sqrt(SIGMAYHAT),N=NMC)+(1-SIZE);
    if(PU < 1){
      CU <- CHYRN(MU=MUGRIDSU,Q=PU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[2] <- 
        ETRN2(MU=MUGRIDSU,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PU >= 1){
      INTERMEDIATE[2] <- 2;
    };
  };
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;
while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  VL <- L.BOUND >= U;
  VU <- U.BOUND <= L;
  INTERMEDIATE <- 2*(VU-VL);
  if(INTERMEDIATE[1] == 0){
    PU <- PTRN2(MU=MUGRIDSL,
                Q=YHAT,
                A=L.BOUND[1],
                B=U.BOUND[1],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)+(1-SIZE);
    if(PU<1){
      CU <- CHYRN(MU=MUGRIDSL,
                  Q=PU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[1] <- 
        ETRN2(MU=MUGRIDSL,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PU >= 1){
      INTERMEDIATE[1] <- 2;
    };
  };
  if(INTERMEDIATE[2] == 0){
    PU <- PTRN2(MU=MUGRIDSM,
                Q=YHAT,
                A=L.BOUND[2],
                B=U.BOUND[2],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)+(1-SIZE);
    if(PU < 1){
      CU <- CHYRN(MU=MUGRIDSM,
                  Q=PU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[2] <- 
        ETRN2(MU=MUGRIDSM,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSM,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PU >= 1){
      INTERMEDIATE[2] <- 2;
    };
  };
  if(INTERMEDIATE[3] == 0){
    PU <- PTRN2(MU=MUGRIDSU,
                Q=YHAT,
                A=L.BOUND[3],
                B=U.BOUND[3],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)+(1-SIZE);
    if(PU<1){
      CU <- CHYRN(MU=MUGRIDSU,
                  Q=PU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[3] <- 
        ETRN2(MU=MUGRIDSU,A=YHAT,B=CU,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PU >= 1){
      INTERMEDIATE[3] <- 2;
    };
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  HYBDCI_UPPER <- MUGRIDSM;
};

CHECK.UNIROOT <- FALSE;
k <- K;

while(CHECK.UNIROOT == FALSE){
  SCALE <- k;
  MUGRIDSL <- YHAT-SCALE*sqrt(SIGMAYHAT);
  MUGRIDSU <- YHAT+SCALE*sqrt(SIGMAYHAT);
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSU));
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  VL <- L.BOUND >= U;
  VU <- U.BOUND <= L;
  INTERMEDIATE <- 2*(VU-VL);
  if(INTERMEDIATE[1] == 0){
    PL <- PTRN2(MU=MUGRIDSL,
                Q=YHAT,A=L.BOUND[1],B=U.BOUND[1],SIGMA=sqrt(SIGMAYHAT),N=NMC)-(1-SIZE);
    if(PL > 0){
      CL <- CHYRN(MU=MUGRIDSL,
                  Q=PL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[1] <- 
        ETRN2(MU=MUGRIDSL,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PL<=0){
      INTERMEDIATE[1] <- -2;
    };
  };
  if(INTERMEDIATE[2] == 0){
    PL <- PTRN2(MU=MUGRIDSU,
                Q=YHAT,
                A=L.BOUND[2],
                B=U.BOUND[2],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)-(1-SIZE);
    if(PL > 0){
      CL <- CHYRN(MU=MUGRIDSU,
                  Q=PL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[2] <- 
        ETRN2(MU=MUGRIDSU,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PL <= 0){
      INTERMEDIATE[2] <- -2;
    };
  };
  HALT.CONDITION <- abs(max(sign(INTERMEDIATE))-min(sign(INTERMEDIATE))) > TOL;
  if(HALT.CONDITION == TRUE){
    CHECK.UNIROOT <- TRUE;
  };
  if(HALT.CONDITION == FALSE){
    k <- 2*k;
  };
};

HALT.CONDITION <- FALSE;

while(HALT.CONDITION == FALSE){
  MUGRIDSM <- (MUGRIDSL+MUGRIDSU)/2;
  MUGRIDS <- c(as.numeric(MUGRIDSL),as.numeric(MUGRIDSM),as.numeric(MUGRIDSU));
  L.BOUND <- pmax(L,MUGRIDS-CV_BETA*sqrt(SIGMAYHAT));
  U.BOUND <- pmin(U,MUGRIDS+CV_BETA*sqrt(SIGMAYHAT));
  VL <- L.BOUND >= U;
  VU <- U.BOUND <= L;
  INTERMEDIATE <- 2*(VU-VL);
  if(INTERMEDIATE[1] == 0){
    PL <- PTRN2(MU=MUGRIDSL,
                Q=YHAT,
                A=L.BOUND[1],
                B=U.BOUND[1],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)-(1-SIZE);
    if(PL > 0){
      CL <- CHYRN(MU=MUGRIDSL,
                  Q=PL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[1] <- 
        ETRN2(MU=MUGRIDSL,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PL <= 0){
      INTERMEDIATE[1] <- -2;
    };
  };
  if(INTERMEDIATE[2] == 0){
    PL <- PTRN2(MU=MUGRIDSM,
                Q=YHAT,
                A=L.BOUND[2],
                B=U.BOUND[2],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)-(1-SIZE);
    if(PL > 0){
      CL <- CHYRN(MU=MUGRIDSM,Q=PL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[2] <- ETRN2(MU=MUGRIDSM,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-ETRN2(MU=MUGRIDSM,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PL <= 0){
      INTERMEDIATE[2] <- -2;
    };
  };
  if(INTERMEDIATE[3] == 0){
    PL <- PTRN2(MU=MUGRIDSU,
                Q=YHAT,
                A=L.BOUND[3],
                B=U.BOUND[3],
                SIGMA=sqrt(SIGMAYHAT),
                N=NMC)-(1-SIZE);
    if(PL > 0){
      CL <- CHYRN(MU=MUGRIDSU,
                  Q=PL,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),CV_BETA=CV_BETA);
      INTERMEDIATE[3] <- 
        ETRN2(MU=MUGRIDSU,A=CL,B=YHAT,SIGMA=sqrt(SIGMAYHAT),N=NMC)-
        ETRN2(MU=MUGRIDSU,A=L,B=U,SIGMA=sqrt(SIGMAYHAT),N=NMC);
    };
    if(PL <= 0){
      INTERMEDIATE[3] <- -2;
    };
  };
  if((abs(INTERMEDIATE[2]) < TOL) || (abs(MUGRIDSU-MUGRIDSL) < TOL)){
    HALT.CONDITION <- TRUE;
  };
  if(sign(INTERMEDIATE[1]) == sign(INTERMEDIATE[2])){
    MUGRIDSL <- MUGRIDSM;
  };
  if(sign(INTERMEDIATE[3]) == sign(INTERMEDIATE[2])){
    MUGRIDSU <- MUGRIDSM;
  };
  HYBDCI_LOWER <- MUGRIDSM;
};

HYBDCI_UPPER <- min(YHAT+CV_BETA*sqrt(SIGMAYHAT),HYBDCI_UPPER);
HYBDCI_LOWER <- max(YHAT-CV_BETA*sqrt(SIGMAYHAT),HYBDCI_LOWER);

rm(YHAT,SIGMAYHAT,L,U,SIZE,NMC,CHECK.UNIROOT,k,SCALE,MUGRIDSL,MUGRIDSU,MUGRIDS,
   INTERMEDIATE,HALT.CONDITION,MUGRIDSM,CV_BETA,PU,CU,PL,CL,VL,VU,L.BOUND,
   U.BOUND);

## -------------------------------------------------------------------------- ##
## PROJECTION CI
## -------------------------------------------------------------------------- ##

YHAT <- YTILDE;
SIGMAYHAT <- SIGMAYTILDE;
CV_BETA <- CV_UNIF;

PROJCI_UPPER <- YHAT+CV_BETA*sqrt(SIGMAYHAT);
PROJCI_LOWER <- YHAT-CV_BETA*sqrt(SIGMAYHAT);

rm(YHAT,SIGMAYHAT,CV_BETA);

## -------------------------------------------------------------------------- ##
## OUTPUT 
## -------------------------------------------------------------------------- ##

OUTPUT <- list("WINNER"=THETA_TILDE,
               "YHAT"=YTILDE,
               "ALPHA"=ALPHA,
               "BETA"=BETA,
               "MEDIAN UNBIASED ESTIMATE"=MED_U_ESTIMATE,
               "OPTIMAL CONDITIONAL CI"=c(CONDCI_LOWER,CONDCI_UPPER),
               "EQUAL-TAILED CONDITIONAL CI"=c(EQUICI_LOWER,EQUICI_UPPER),
               "HYBRID ESTIMATE"=MED_HYBD_ESTIMATE,
               "HYBRID CI"=c(HYBDCI_LOWER,HYBDCI_UPPER),
               "EQUAL-TAILED HYBRID CI"=c(EQUICI_HYBD_LOWER,EQUICI_HYBD_UPPER),
               "PROJECTION CI"=c(PROJCI_LOWER,PROJCI_UPPER));
# COLLECT THE OUTPUT OF THE FUNCTION.

rm(MED_U_ESTIMATE,CONDCI_LOWER,CONDCI_UPPER,EQUICI_LOWER,EQUICI_UPPER,
   MED_HYBD_ESTIMATE,HYBDCI_LOWER,HYBDCI_UPPER,EQUICI_HYBD_LOWER,
   EQUICI_HYBD_UPPER,PROJCI_LOWER,PROJCI_UPPER);
# RESTORE GLOBAL ENVIRONMENT TO ORIGINAL STATE.

print(OUTPUT);
# PRINT THE OUTPUT.

## -------------------------------------------------------------------------- ##
## UNDO PRELIMINARIES
## -------------------------------------------------------------------------- ##

rm(ETRN2,PTRN2,CUTRN,CHYRN);
# REMOVE DEFINED FUNCTIONS.

rm(TOL,K,THETA_TILDE,YTILDE,SIGMAY,SIGMAYTILDE,SIGMAXYTILDE_VEC,SIGMAXYTILDE,
   ZTILDE,IND_L,IND_U,IND_V,LTILDE,UTILDE,VTILDE,GAMMA,M.FACTORS,GAUSSIAN_MAX,
   CV_UNIF_BETA,CV_UNIF);
# RESTORE GLOBAL ENVIRONMENT TO ORIGINAL STATE.