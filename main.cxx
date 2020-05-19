#include <iostream>
#include <fstream>

#include "../eigen/Eigen/Dense"



void export_variables(Eigen::MatrixXd& U,
		      Eigen::VectorXd& A,
		      int step){
  std::ofstream out;
  out.open("results/step"+std::to_string(step)+".dat");
  int nx = U.rows();
 
  double gamma = 1.4 ;
  double R = 8.314; 
  double mw_air = 28.97/1000.;// molecular weight of air [kg/mol]
  // Derived thermo properties
  double Rair = R/mw_air;// Mass ideal gas constant for air [J/(kg*K)]
  //double patm = 101325;
  double patm = 11800 ;
  for(int i=0; i<nx; ++i){
    double Ai = A(i);
    double rho = U(i,0)/Ai;
    double u   = U(i,1)/U(i,0) ;
    double rho_E = U(i,2)/Ai; 
    double p = (gamma-1)*(rho_E-0.5*rho*u*u);
    double T = p/(rho*Rair); 
    out<< 3*i/double(nx) <<" " << rho <<" "<< u/340. <<" " << T <<" " <<p<< std::endl; 
  }
  out.close(); 
}

int main(){

  
  double gamma = 1.4;// cp/cv in ideal gas for air         
  double mw_air = 28.97/1000.;// molecular weight of air [kg/mol]
  double R = 8.314;      
  // Derived thermo properties
  double Rair = R/mw_air;// Mass ideal gas constant for air [J/(kg*K)]
  double cv = Rair/(gamma-1);// specific heat of air at constant volume [J/(kg*K)]


  
  double L = 3; 
  int nx = 200;
  double cfl = 0.8;// CFL number to choose time step
  // Define Quasi-1D geometry
  double dx = L/double(nx) ;
 
  
  // define nozzle geometry [xi, Ai, dAi/dx]
  Eigen::VectorXd A(nx+1); 
  for(int i=0; i<nx+1; ++i){    
    double xi = i*dx ; 
    A(i) = 1+15*std::pow(xi/L -0.5,2);
  }


  // Input parameters  
  double patm = 101325;//[Pa]
  double Tatm = 300;//[K]


  // 3 var [rhoA, rhoAu, eA]
  // +2 flow [A(rhoAu2+p), u(e+p)*A]
  Eigen::MatrixXd U(nx+1, 5) ;
  U.setZero();
  /*
  double pres = 100*patm;
  double Tres = 1273;
  double aref = std::sqrt(gamma*Rair*Tres);
  for(int i=0; i<nx+1; ++i){
    double xi = i*dx ;
    double pi = ((pres-patm)*exp(-xi/(L/15)) + patm); 
    double Ti = ((Tres-Tatm)*exp(-xi/(L/15)) + Tatm); 
    double vi = 1.e-3*aref; 
    double& Ai = A(i); 
    double rhoi = pi/(Ti*Rair);    
    double ei = Rair*Ti/(gamma-1);
    U(i,0) = rhoi*Ai;
    U(i,1) = rhoi*Ai*vi;   
    U(i,2) = rhoi*Ai*(ei+0.5*vi*vi) ;
  }
  */

  double Tref = 232; 
  double aref = std::sqrt(gamma*Rair*Tref);
  double pref = 11800 ;
  double vref = 340;  
  std::cout<<" aref: "<< aref << std::endl ; 
  for(int i=0; i<nx+1; ++i){
    double pi = pref ;
    double Ti = Tref;
    double xi = i*dx/L; 
    double vi = (vref +xi*vref); 
    double& Ai = A(i); 
    double rhoi = pi/(Ti*Rair);    
    double ei = Rair*Ti/(gamma-1);
    U(i,0) = rhoi*Ai;
    U(i,1) = rhoi*Ai*vi;   
    U(i,2) = rhoi*Ai*(ei+0.5*vi*vi) ;
  }

  // Main Loop
  Eigen::MatrixXd Up(nx+1, 5) ;//predictor values
  Up.setZero(); 


  double time = 0.0;
  double t_end = 0.1;

  for(int step = 0; step<10000; ++step){

    double dt = 1.e-5;
    // find time step
    for(int i=0; i<nx+1; ++i){
      double u = U(i,1)/U(i,0); 
      double rho = U(i,0)/A(i);
      double rho_E = U(i,2)/A(i); 
      double p = (gamma-1)*(rho_E-0.5*rho*u*u);
      double c = std::sqrt(gamma*p/rho); 
      double dti = cfl*dx/(std::fabs(u) +c); 
      if(dti<dt){
	dt  =dti;
      }
    }
    
    double h = dt/dx;
    double h2 = 0.5*h ;
  
    // Update U flux variables (conservative)
    for(int i=0; i<nx+1; ++i){ 
      double u2 = 0.5*U(i,1)*U(i,1)/U(i,0);      
      U(i,3) = (3-gamma)*u2 + (gamma-1)*U(i,2); 
      U(i,4) = (gamma*U(i,2)-(gamma-1)*u2)*U(i,1)/U(i,0); 
    }

    // Predictor
    for(int i=1; i<nx; ++i){
      double u2 =  0.5*U(i,1)*U(i,1)/U(i,0);   
      double p = (gamma-1)*(U(i,2)-u2)/A(i);
      Up(i,0) = U(i,0) - h*(U(i+1,1)-U(i,1)) ;
      Up(i,1) = U(i,1) - h*(U(i+1,3)-U(i,3)) + h*p*(A(i+1)-A(i)); 
      Up(i,2) = U(i,2) - h*(U(i+1,4)-U(i,4)) ;
    }
    for(int j=0; j<3; ++j) Up(0, j)  = U(0, j) ;
    for(int j=0; j<3; ++j) Up(nx, j) = U(nx, j) ;

    // Update Up flux variables
    for(int i=0; i<nx+1; ++i){
      double u2 = Up(i,1)*Up(i,1)/Up(i,0);      
      Up(i,3) = 0.5*(3-gamma)*u2 + (gamma-1)*Up(i,2); 
      Up(i,4) = (gamma*Up(i,2)-0.5*(gamma-1)*u2)*Up(i,1)/Up(i,0); 
    }

    // Corrector
    double unorm = 0 ;
    for(int i=1; i<nx; ++i){ 
      double u2 =  0.5*Up(i,1)*Up(i,1)/Up(i,0);   
      double p = (gamma-1)*(Up(i,2)-u2)/A(i);
      U(i,0) = 0.5*(U(i,0)+Up(i,0)) - h2*(Up(i,1)-Up(i-1,1)) ;
      U(i,1) = 0.5*(U(i,1)+Up(i,1)) - h2*(Up(i,3)-Up(i-1,3)) + h2*p*(A(i)-A(i-1)); 
      U(i,2) = 0.5*(U(i,2)+Up(i,2)) - h2*(Up(i,4)-Up(i-1,4)) ;
      unorm += U(i,1)*U(i,1)/U(i,0) ;
    }
    
    // inlet bc
    U(0,1) = 2*U(1,1)-U(2,1); 
    // outlet bc
    for(int j=0; j<3; ++j) U(nx,j) = 2*U(nx-1,j) - U(nx-2,j);

    export_variables(U, A, step); 

    if(step%100 ==0){
      std::cout<<" step:"<< step<<", t: "<<time <<" dt: "<< dt <<" u2: "<< unorm<< std::endl; 
      
    }
    
    time += dt; 
  }

  

  return 0;
}; 
