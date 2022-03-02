% Source code for our paper 10.13195/j.kzyjc.2021.0292
% Toolbox openVectorField (https://github.com/wjxjmj/openVectorField) is needed.
global n
para.dim=2;
n=10; %
m=randi(1,1,n);
stime=60;
ICL_n=60;
N=1000;
para.ICL_n=ICL_n;
para.gamma=0.1;
para.m=m; %
para.m0=28;
para.g=9.8;
para.n=n;
para.d_dim=3;
para.s=unifrnd(1,3,[para.n,1]);
para.omega=unifrnd(-1,1,[para.n,1]);
if para.dim==2
    para.D=[1,0,0;0,1,0];
elseif para.dim==3
    para.D=diag([1,1,1]);
end
para.rr=5;
para.r=1.2; %
para.la=1; %
para.a=50; %
para.b=1; %
para.c=1;
para.rs=2;  
para.fcl_k=1;
para.fcl_l=1;
para.rcl_a=0;
para.ficl_a=1;
para.stime=stime;
para.N=N;
para.rr=5;
para.rh=5;

para.enableDataConsensus=0;
para.enableDisturbances=0;
para.enableCL=1;
para.enableICL=1;
para.enableParameterConsensus=1;   
para.enablePinv=0;

x0=zeros(para.dim,1);
theta=0.5*pi;
[y0,th0]=f(1,x0,x0,x0,x0,para);
[y1,th1]=f1(1,x0,x0,para);
[y2,th2]=f2(1,x0,x0,x0,para);

state1=[];
if para.dim==3
    state1.q = unifrnd(-8,8,[para.dim,n]);
    state1.p = unifrnd(-1,1,[para.dim,n]);
elseif para.dim==2
    state1.q = unifrnd(0,6,[para.dim,n]);
    state1.p = unifrnd(-1,1,[para.dim,n]);
end
if para.dim==3
    state1.q(3,:)=zeros(1,para.n);
    state1.p(3,:)=zeros(1,para.n);
end
state1.d = unifrnd(-2,2,[para.d_dim,para.n]);
% state1.xd=unifrnd(-5,5,[dim,1])+[0;0;0];

% state1.vd=[cos(theta),sin(theta),0;-sin(theta),cos(theta),0;0,0,1]*state1.xd;
state1.th = unifrnd(-0,0,[length([th1;th2]),n]);
% state.th = repmat(th0,[1,para.n]);


state2=state1;
state2.hat_vd = zeros([para.dim,para.n]);
state2.alpha=zeros(para.n,para.n+1);
state2.beta =zeros(1,para.n);

state1.kc = zeros([1,para.n]);

sim1 = vectorField(state1,@(t,state)adaptiveSmcFlocking(t,state,para));
sim1.solve('ode113',linspace(0,para.stime,1000),state1);
[t1,data1]=sim1.result(para.N);

sim2 = vectorField(state2,@(t,state)comparativeFlocking(t,state,para));
sim2.solve('ode113',linspace(0,para.stime,1000),state2);
[t2,data2]=sim2.result(para.N);

figure(1)
plotFormation_t(t1,data1,'proposed',[1,para.stime*90/240,para.stime*160/240,para.stime],para)
figure(2)
plotSim(t1,data1,'AC',para)
figure(3)
plotPara(t1,data1,'AC',para)
figure(4)
plotFormation_t(t2,data2,'proposed',[1,para.stime*90/240,para.stime*160/240,para.stime],para)
figure(5)
plotSim(t2,data2,'AC',para)
figure(6)
plotPara(t2,data2,'AC',para)



% figure(1)
% if para.dim==3
% plotGif(t1,data1,'AC',para)
% axis equal
% elseif para.dim==2
%     plotFormation_t(t1,data1,'AC',[1,80,130,200],para)
% end
% % plotFormation(t1,data1,'AC',para)
% figure(2)
% plotSim(t1,data1,'AC',para)
% figure(3)
% plotPara(t1,data1,'AC',para)
% figure(4)
% plot(t1,data1.kc,'b-')



function grad = adaptiveSmcFlocking(t,state,para)
x=state.q;
v=state.p;
th=state.th;
d = state.d;

% generate trajectory
[xd,vd,ad] = virtualLeader(t,para);

% generate disturbances
a=35;b=3;c=28;
dot_d=[a*(d(2,:)-d(1,:));
     (c-a).*d(1,:)-d(1,:).*d(3,:)+c*d(2,:);
      d(1,:).*d(2,:)-b*d(3,:)];


gamma=para.gamma;

u=zeros(size(state.p));
[phi_id,dot_phi_id]=leaderForce(x,v,xd,vd,para.rs,para.c);
[phi_ij,dot_phi_ij]=agentForce(x,v,para.r,para.la,para.a,para.b);
u_dot_q=zeros(size(state.p));
u_th=zeros(size(state.th));
u_kc=zeros(size(state.kc));

for i=1:para.n
    qi=x(:,i);
    dot_qi=v(:,i);
    thi=th(:,i);
    ri = phi_id(:,i)+phi_ij(:,i)-vd;
    dri =1*dot_phi_id(:,i)+1*dot_phi_ij(:,i)-ad;
    ei=ri+dot_qi;
    [z,~]=f(i,qi,dot_qi,-dri,-ri,para);
    [z2,a2]=f2(i,qi,dot_qi,dot_qi,para);
    u_th(:,i)=-gamma*z'*ei;
%     u(:,i) = z*thi-10*ei-10*(dot_qi-vd)-0.*state.kc(i)*signk(ei,0.01);
%     u_dot_q(:,i) = 1/para.m(i)*(u(:,i)+0.0*para.D*d(:,i)-z2*a2);
%     u_kc(i)=0*norm(ei);
    u(:,i) = z*thi-10*ei-10*(dot_qi-vd)-1.*state.kc(i)*sign(ei);
    u_dot_q(:,i) = 1/para.m(i)*(u(:,i)+1*para.D*d(:,i)-z2*a2);
%     u(:,i) = z*thi-10*ei-10*(dot_qi-vd)-2.*state.kc(i)*signk(ei,0.001);
%     u_dot_q(:,i) = 1/para.m(i)*(u(:,i)+0.1*para.D*d(:,i)-z2*a2);
    u_kc(i)=norm(ei);
end
grad.q=v;
grad.p=u_dot_q;
grad.th=u_th;
grad.d = dot_d;
grad.kc = u_kc;
end

function grad = comparativeFlocking(t,state,para)
x=state.q;
v=state.p;
th=state.th;
d = state.d;
alpha = state.alpha;
beta = state.beta;
hat_vd = state.hat_vd;

% generate trajectory
[xd,vd,ad] = virtualLeader(t,para);

% generate disturbances
a=35;b=3;c=28;
dot_d=[a*(d(2,:)-d(1,:));
    (c-a).*d(1,:)-d(1,:).*d(3,:)+c*d(2,:);
    d(1,:).*d(2,:)-b*d(3,:)];


Gamma=para.gamma;

u=zeros(size(state.p));
% [phi_id,dot_phi_id]=leaderForce(x,v,xd,vd,para.rs,para.c);
% [phi_ij,dot_alpha]=agentForce(x,v,xd,vd,para.r,alpha,para.la,para.a,para.b,para.gamma1);
u_dot_q=zeros(size(state.p));
u_th=zeros(size(state.th));
dot_alpha = zeros(size(alpha));
u_beta=zeros(size(beta));
u_hat_vd = zeros(size(hat_vd));
for i=1:para.n
    qi=x(:,i);
    dot_qi=v(:,i);
    thi=th(:,i);
    xi0 = qi-xd;
    vi0=dot_qi-vd;
    
    u1_i=zeros(size(qi));
    si=dot_qi-hat_vd(:,i);
    for j=1:para.n
        if i==j
            continue
        else
            qj=x(:,j);
            dot_qj=v(:,j);
            xij=qi-qj;
            vij=dot_qi-dot_qj;
            dis = norm(qi-qj);
            if dis<para.r
                aij   = rho_h(dis^2,para.r^2,0.8);
                dir   = xij/(dis+0.0001);
                u1_i = u1_i+aij*phi(dis,para.la,para.r,para.a,para.b)*dir;
                u1_i = u1_i+alpha(i,j)*aij*sign(vij);
%                 dot_alpha(i,j)=para.gamma1(i)*aij*norm(vi0,1);
            end
        end
        u1_i = u1_i+ xi0 + vi0 ;
%         u1_i = u1_i+(qi - xd) +alpha(i,para.n+1)*vi0;
    end
    u_hat_vd(:,i) = -10*u1_i - 5.0*sign(si);
    hat_u_i = -10*u1_i ;%- beta(i)*(si);
    [Yi,~]=f(i,qi,dot_qi,u_hat_vd(:,i),hat_vd(:,i),para);
    [z2,a2]=f2(i,qi,dot_qi,dot_qi,para);
    
    u_i = hat_u_i + Yi * thi;
    
    u_th(:,i)=-Gamma*Yi'*si;
    u(:,i) = u_i;
    u_dot_q(:,i) = 1/para.m(i)*(u(:,i)+1*para.D*d(:,i)-z2*a2);
%     u_beta(i)=para.gamma2(i)*norm(si,1);
end
grad.q=v;
grad.p=u_dot_q;
grad.th=u_th;
grad.d = dot_d;
grad.alpha = dot_alpha;
grad.beta = u_beta;
grad.hat_vd = u_hat_vd;
end

function [xd,vd,ad] = virtualLeader(t,para)
if para.dim==3
d_angle = 4*pi/para.stime;
angle = t * d_angle;
rs = para.rr;
rh = para.rh;
d_height = para.rh/para.stime;
height = t * d_height;
xd=[cos(angle)*rs;sin(angle)*rs;height];
vd=[-sin(angle)*d_angle*rs;cos(angle)*d_angle*rs;d_height];
ad=[-cos(angle)*(d_angle)^2*rs;-sin(angle)*(d_angle)^2*rs;0];
elseif para.dim==2
rr = para.rr;
xd = [cos(t*1.8*pi/para.stime)*rr;
      sin(t*1.8*pi/para.stime)*rr];
vd = [-sin(t*1.8*pi/para.stime)*rr*1.8*pi/para.stime;
      cos(t*1.8*pi/para.stime)*rr*1.8*pi/para.stime];
ad = [-cos(t*1.8*pi/para.stime)*rr*(1.8*pi/para.stime)^2;
      -sin(t*1.8*pi/para.stime)*rr*(1.8*pi/para.stime)^2];
end
end

function y=chief(xd,vd)
g=9.8;
m0=1;
% y=-1/m0*g/norm(xd)^3*xd;
y=-0.1*xd;
end


function [y,a]=f(i,x,v,alpha,beta,para)
[y1,a1]=f1(i,x,alpha,para);
[y2,a2]=f2(i,x,v,beta,para);
y=[y1,y2];
a=[a1;a2];
% r0=norm(xd);
% ri=norm([r0;0;0]+x);
% y=zeros(para.dim,para.dim);
% y(:,1)=alpha;
% y(1,2)=-2*beta(2)/r0^(1.5)+3*x(2)/2/r0^(2.5)*xd'*vd/r0;
% y(2,2)= 2*beta(1)/r0^(1.5)-3*x(1)/2/r0^(2.5)*xd'*vd/r0;
% y(1,3)=-1/r0^3*x(1)+(r0+x(1))/ri^3-1/r0^2;
% y(2,3)=-1/r0^3*x(2)+x(2)/ri^3;
% y(3,3)=x(3)/ri^3;
% a=para.m.*[1;para.g^0.5;para.g];
end

function [y,a]=f1(i,x,alpha,para)
y=alpha;
a=[para.m(i)];
end

function [y,a]=f2(i,x,v,beta,para)
if para.dim==3
y=zeros(3,3);
y(1,1)=-v'*v;
y(2,1)=-v'*v;
y(3,1)=-v'*v;
y(1,2)=-beta(2);
y(2,2)= beta(2);
y(1,3)=1;
y(2,3)=1;
y(3,3)=1;
a=[para.s(i);para.m(i)*para.omega(i);para.m(i)*para.g];
elseif para.dim==2
y=zeros(2,3);
y(1,1)=-v'*v;
y(2,1)=-v'*v;
y(1,2)=-beta(2);
y(2,2)= beta(2);
y(1,3)=1;
y(2,3)=1;
a=[para.s(i);para.m(i)*para.omega(i);para.m(i)*para.g];   
end

end


function [y,dy]=leaderForce(x,v,xd,vd,rs,c)
y  = c.* (x-repmat(xd,[1,size(x,2)]));
dy = c.* (v-repmat(vd,[1,size(v,2)]));
end

% function [y,dy]=leaderForce(x,v,xd,vd,rs)
% c=10;
% e=0.01;
% [d,n]=size(x);
% y=zeros(size(x));
% dy=zeros(size(x));
%
% for i=1:n
%     [p,dp]=proj(x(:,i),v(:,i),xd,vd,rs);
%     [yi,dyi]=attr(x(:,i),v(:,i),p,dp,rs);
%     y(:,i)=c.*(x(:,i)-p);
%     dy(:,i)=c.*(v(:,i)-dp);
% end
% end


function [y,dy]=agentForce2(x,v,r,la,a,b)
e=0.01;
n = size(x,2);
y = zeros(size(v));
dy= zeros(size(v));
for i=1:n
    xi=x(:,i);
    vi=v(:,i);
    for j=1:n
        if i==j
            continue
        else
            xj=x(:,j);
            vj=v(:,j);
            xij=xi-xj;
            vij=vi-vj;
            dis = norm(xi-xj);
            if dis<r
                g=b;
                if dis>la
                    g=a;
                end
                aij   = rho_h(dis^2,r^2,0.8);
                d_aij = d_rho_h(dis^2,r^2,0.8)*2*xij'*vij;
                arij  = g*((dis^2-la^2)*pi);
                d_arij= g*pi*2*xij'*vij;
                dir   = xij/(dis+e);
                d_dir = (-1/(dis*(e+dis)^2).*xij*xij'+1/(e+dis).*eye(size(xij,1)))*vij;
                y(:,i) = y(:,i);
                dy(:,i)=dy(:,i)+d_aij*arij*dir+aij*d_arij*dir+aij*arij*d_dir;
            end
        end
    end
end
end

function [y,dy]=agentForce(x,v,r,la,a,b)
e=0.01;
n = size(x,2);
y = zeros(size(v));
dy= zeros(size(v));
for i=1:n
    xi=x(:,i);
    vi=v(:,i);
    for j=1:n
        if i==j
            continue
        else
            xj=x(:,j);
            vj=v(:,j);
            xij=xi-xj;
            vij=vi-vj;
            dis = norm(xi-xj);
            if dis<r
                g=b;
                if dis>la
                    g=a;
                end
                aij   = rho_h(dis^2,r^2,0.8);
                d_aij = d_rho_h(dis^2,r^2,0.8)*2*xij'*vij;
                arij  = g*((dis^2-la^2)*pi);
                d_arij= g*pi*2*xij'*vij;
                dir   = xij/(dis+e);
                d_dir = (-1/(dis*(e+dis)^2).*xij*xij'+1/(e+dis).*eye(size(xij,1)))*vij;
                y(:,i) = y(:,i)+phi(dis,la,r,a,b)*dir;
                dy(:,i)=dy(:,i)+dPhi(dis,la,r,a,b)*xij'*vij/dis*dir+phi(dis,la,r,a,b)*d_dir;
            end
        end
    end
end
end

function y=smoothUp(x)
if x>1
    y=1;
elseif x<0
    y=0;
else
    y=x-sin(2*pi*x)/(2*pi);
end
end
function y=smoothDown(x)
if x>1
    y=0;
elseif x<0
    y=1;
else
    y=1-x+sin(2*pi*x)/(2*pi);
end
end
function y=smoothUpDown(x,h1,h2)
if x<h1
    y=smoothUp(x/h1);
elseif x<h2
    y=1;
else
    y=smoothDown((x-h2)/(1-h2));
end
end

function y=phi(x,l,r,a,b)
h1=0.3;
h2=0.7;
y=-a*smoothUpDown(x/l,h1,h2)+b*smoothUpDown((x-l)/(r-l),h1,h2);
end

function y=dPhi(x,l,r,a,b)
h1=0.3;
h2=0.7;
y=-a/l*dSmoothUpDown(x/l,h1,h2)+b/(r-l)*dSmoothUpDown((x-l)/(r-l),h1,h2);
end

function y=phid(x)
h=1.5;
if x<1
    y=0;
elseif x<h
    y=smoothUp((x-1)/(h-1));
else
    y=1;
end
end

function y=dPhid(x)
h=1.5;
if x<1
    y=0;
elseif x<h
    y=1/(h-1)*dSmoothUp((x-1)/(h-1));
else
    y=0;
end
end

function y=dSmoothUp(x)
if x>1
    y=0;
elseif x<0
    y=0;
else
    y=2*sin(pi*x)^2;
end
end
function y=dSmoothDown(x)
if x>1
    y=0;
elseif x<0
    y=0;
else
    y=-1+cos(2*pi*x);
end
end
function y=dSmoothUpDown(x,h1,h2)
if x<h1
    y=1/h1*dSmoothUp(x/h1);
elseif x<h2
    y=0;
else
    y=1/(1-h2)*dSmoothDown((x-h2)/(1-h2));
end
end

function y=rho_h(z,r,h)
if z<h*r && z>=0
    y=1;
elseif z<=r && z>=h*r
    y=(2*pi*(-r+z)+(-1+h)*r*sin(2*pi*(-h*r+z)/(r-h*r)))/(2*(h-1)*r*pi);
else
    y=0;
end
end

function y=d_rho_h(z,r,h)
if z<h*r && z>=0
    y=0;
elseif z<=r && z>=h*r
    y=2*sin(pi*(-h*r+z)/(r-r*h))^2/((-1+h)*r);
else
    y=0;
end
end

function plotSim(t,data,methodName,para)

vd=zeros(length(t),para.dim);
for i=1:length(t)
    [~,vd_t,~] = virtualLeader(t(i),para);
    vd(i,:)=vd_t';
end
data.vd=vd;

plot(t,data.p-repmat(data.vd,[1,para.n]))

% dim=length(data.xd(end,:));
% plot3(data.q(:,1:dim:end),data.q(:,2:dim:end),data.q(:,3:dim:end));hold on
% plot3(data.xd(:,1:dim:end),data.xd(:,2:dim:end),data.xd(:,3:dim:end),'r--');
% plot3(data.q(end,1:dim:end),data.q(end,2:dim:end),data.q(end,3:dim:end),'o');
% plot3(data.xd(end,1:dim:end),data.xd(end,2:dim:end),data.xd(end,3:dim:end),'rp');
title(['velocity mismatch (',methodName,')'])
grid on
% axis equal
end


function plotFormation(t,data,methodName,para)

dim=length(data.xd(end,:));
plot3(data.q(:,1:dim:end),data.q(:,2:dim:end),data.q(:,3:dim:end));
plot3(data.xd(:,1:dim:end),data.xd(:,2:dim:end),data.xd(:,3:dim:end),'r--');
plot3(data.q(end,1:dim:end),data.q(end,2:dim:end),data.q(end,3:dim:end),'o');hold on
plot3(data.xd(end,1:dim:end),data.xd(end,2:dim:end),data.xd(end,3:dim:end),'rp');
n=numel(data.q(end,:))/dim;
q=reshape(data.q(end,:),[dim,n]);
for i=1:n
    qi=q(:,i);
    for j=1:n
        qj=q(:,j);
        if i==j
            continue
        else
            dis=norm(qi-qj);
            if dis<para.la*0.95
                line([qi(1);qj(1)],[qi(2);qj(2)],[qi(3);qj(3)],'color','r')
            elseif dis<para.r
                line([qi(1);qj(1)],[qi(2);qj(2)],[qi(3);qj(3)],'color','b')
            end
        end
    end
end
hold off
title(['formation (',methodName,')'])
grid on
axis equal
end

function plotPara(t,data,methodName,para)
th_star=[];
for i=1:para.n
    th_star=[th_star,para.m(i),para.s(i),para.m(i)*para.omega(i),para.m(i)*para.g];
end
th_star=repmat(th_star,[length(t),1]);

th_delta = data.th-th_star;
th_delta_end=[data.th(end,:);th_star(end,:)];
plot(t,th_delta);

% plot(t,data.th)
% hold on
% th_line=[repmat(para.m,[length(t),1]),ones(length(t),1).*para.g^0.5,ones(length(t),1).*para.g];
% plot(t,th_line,'--')
% hold off
title(['parameters estimation (',methodName,')'])
end
function plotGif(t,data,methodName,para)

xd=zeros(length(t),3);
for i=1:length(t)
    [xd_i,~,~]=virtualLeader(t(i),para);
    xd(i,:)=xd_i';
end
data.xd=xd;

dim=length(data.xd(end,:));
plot3(data.q(1,1:dim:end),data.q(1,2:dim:end),data.q(1,3:dim:end),'x');hold on
% for i=2:10:length(t)
%     plot3(data.xd(1:i,1:dim:end),data.xd(1:i,2:dim:end),data.xd(1:i,3:dim:end),'r--');
%     plot3(data.q(1:i,1:dim:end),data.q(1:i,2:dim:end),data.q(1:i,3:dim:end),'b-');
%     drawnow
% end
plot3(data.xd(1:end,1:dim:end),data.xd(1:end,2:dim:end),data.xd(1:end,3:dim:end),'r--');
plot3(data.q(1:end,1:dim:end),data.q(1:end,2:dim:end),data.q(1:end,3:dim:end),'c-');
plot3(data.q(end,1:dim:end),data.q(end,2:dim:end),data.q(end,3:dim:end),'bo');
plot3(data.xd(end,1:dim:end),data.xd(end,2:dim:end),data.xd(end,3:dim:end),'rp');
% plotCircle(data.xd(end,:),para.rs)

n=numel(data.q(end,:))/dim;
q=reshape(data.q(end,:),[dim,n]);
for i=1:n
    qi=q(:,i);
    for j=1:n
        qj=q(:,j);
        if i==j
            continue
        else
            dis=norm(qi-qj);
            if dis<para.la*0.9
                line([qi(1);qj(1)],[qi(2);qj(2)],[qi(3);qj(3)],'color','r')
            elseif dis<para.r
                line([qi(1);qj(1)],[qi(2);qj(2)],[qi(3);qj(3)],'color','b')
            end
        end
    end
end
hold off
title(['Trajectory (',methodName,')'])
grid on
% axis equal
end
function y=signs(x)
p=0.01;
y=sign(x).*abs(x).^p;
end
function y=signk(x,p)
if x<=p
    y=-ones(size(p));
elseif x>p
    y=ones(size(p));
else
    y=1/p.*x;
end
end


function plotFormation_t(t,data,methodName,ts,para)

dim=para.dim;

% xd = zeros(length(t),para.dim);
xd = [cos(t*1.8*pi/para.stime)*para.rr,...
      sin(t*1.8*pi/para.stime)*para.rr];

% plot(data.q(:,1:dim:end),data.q(:,2:dim:end));
q = data.q;
plot(xd(:,1:dim:end),xd(:,2:dim:end),'r--','linewidth',2);hold on

plot(q(1,1:dim:end),q(1,2:dim:end),'kx');

for k=1:length(ts)
    ts_i = ts(k);
    t_delta = t - ts_i;
    t_delta = abs(t_delta);
    [~,index]=min(t_delta);
    t_i = index;
    plotFormationAtTime(data.q(t_i,:),xd(t_i,:),para)
end

plotFormationAtTime(data.q(end,:),xd(end,:),para)

hold off
title(['formation (',methodName,')'])
grid on
axis equal
end

function plotFormationAtTime(q,xd,para)
dim=para.dim;
n = para.n;

q_mat=reshape(q,[dim,n]);
for i=1:n
    qi=q_mat(:,i);
    for j=1:n
        qj=q_mat(:,j);
        if i==j
            continue
        else
            dis=norm(qi-qj);   
            if dis<para.la*0.95
                line([qi(1);qj(1)],[qi(2);qj(2)],'color',[0.6,0.6,0.6])
            elseif dis<para.r
                line([qi(1);qj(1)],[qi(2);qj(2)],'color',[0.6,0.6,0.6])
            end
        end
    end
end
plot(q(1:dim:end),q(2:dim:end),'o','Color',[1,1,1],'MarkerFaceColor',[0.2,0.6,1]);
plot(xd(1:dim:end),xd(2:dim:end),'p','Color','r','MarkerFaceColor','r');
end