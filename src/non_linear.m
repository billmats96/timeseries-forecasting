%% Study B
close all
clear 

data = importdata('dat4.dat');
week_data = data(6:6:end);
detrended = diff(week_data,1);
window_week = length(detrended);

alpha = 0.05;
maxtau = 20;
zalpha = norminv(1-alpha/2);
autlim = zalpha/sqrt(window_week);

Tmax = 10;
max_p = 5;
max_q = 5;
aic_fit = zeros(max_p+1, max_q+1);
nrmse_fit = cell(max_p+1, max_q+1);

% Visual representation of data
figure('name', 'Week_data');
g = suptitle({['Week Data'],' ',' '});
set(g, 'FontSize', 12, 'FontWeight', 'bold')
subplot(231);
plot(week_data)
ylabel('Exchange rate'); 
xlabel('Sample')
title('Original Time Series')

% Trend removal using first differences on time series      
subplot(232);
plot(detrended)
ylabel('Exchange rate'); 
xlabel('Sample')
title('Detrended Time Series (First Diffs)'); 

% Stationary time series autocorellation & partial autocorellation 
acM = autocorrelation(detrended, maxtau);
subplot(233);
hold on

for ii = 1:maxtau
    plot(acM(ii+1,1)*[1 1],[0 acM(ii+1,2)],'b','linewidth',1.5)
end

plot([0 maxtau+1],[0 0],'k','linewidth',1.5)
plot([0 maxtau+1],autlim*[1 1],'--c','linewidth',1.5)
plot([0 maxtau+1],-autlim*[1 1],'--c','linewidth',1.5)
xlabel('\tau')
ylabel('r(\tau)')
title('Stationary Time Series Autocorrelation'); 
important_autocorrs = length(find(abs(acM(:,2))>autlim)) - 1; %sub 1 because at lag 0, it's always 1
hold off

pacfV = parautocor(detrended, maxtau);
subplot(234)
hold on

for ii=1:maxtau
    plot(acM(ii+1,1)*[1 1],[0 pacfV(ii)],'b','linewidth',1.5)
end

plot([0 maxtau+1],[0 0],'k','linewidth',1.5)
plot([0 maxtau+1],autlim*[1 1],'--c','linewidth',1.5)
plot([0 maxtau+1],-autlim*[1 1],'--c','linewidth',1.5)
xlabel('\tau')
ylabel('\phi_{\tau,\tau}')
title('Stationary Time Series Partial Autocorrelation');

% Portmanteau Test
acM = autocorrelation(detrended, maxtau);
tittxt = sprintf('Ljung-Box Test');
subplot(235)
[h2V,p2V,Q2V] = portmanteauLB(acM(2:maxtau+1,2),window_week,alpha);
plot([1:maxtau]',p2V,'.-k')
hold on
plot([0 maxtau+1],alpha*[1 1],'--c')   
xlabel('lag \tau')
ylabel('p-value')
title(' Ljung-Box Portmanteau test')
axis([0 maxtau+1 0 1])

% Fit of an ARMA model
if(sum(h2V) > 1)

    for p = 0:max_p
        for q = 0:max_q
            [nrmseV,phiallV,thetaallV,SDz,aicS,fpeS] = fitARMA(detrended,p,q,Tmax);
            aic_fit(p+1,q+1) = aicS; 
            nrmse_fit{p+1,q+1} = nrmseV;
        end
    end

     min_aic = min(min(aic_fit));
    [min_aic_p, min_aic_q] = find(aic_fit == min_aic);

    fprintf('Window_week: \n');
    fprintf('\t T \t\t NRMSE \n');
    disp([[1:Tmax]' nrmse_fit{min_aic_p, min_aic_q}])


    subplot(236)
    hold on

    for p = 0:max_p    
        plot([0 : max_q], aic_fit(p+1, :));
    end

    hold off
    legend('p = 0', 'p = 1', 'p = 2', 'p = 3', 'p = 4', 'p = 5')
    xlabel('q')
    ylabel('AIC(p,q))')
    title('AIC of ARMA Model')
else

    %nrmse_fit{p+1,q+1} = 1;
    fprintf('\n Window_week : White Noise Time Series -> NRMSE = 1. \n');

end

% saveas(gcf, 'Time_Series_Week', 'svg');


%% Prediction (1 step) study B

p = 5;
q = 0;
Tmax = 1 ;
nrmse_vec = zeros(1,2);


% week_data = week_data(1:end-1);
    
n = length(detrended); % time series length   
proptest = 0.4; % proportion of the whole time series to be used as test 100/250

xV = detrended; 
xV_train = xV(1:ceil((1 - proptest)*n)); % training set
xV_test = xV(ceil((1 - proptest)*n+1):end); % testing set
nlast = floor(proptest*n);

error_vec = zeros(nlast,1);

[nrmseV,preM,phiV,thetaV] = predictARMAnrmse(xV,p,q,1,nlast);
predicted = preM + week_data(n-nlast+1:end-1);
figure()
subplot(211);
plot(week_data(n-nlast+1:end), 'color' , 'blue')
hold on
plot(predicted, 'color', 'red')
title('Non stationary time series Prediction ', 'FontSize', 14)
xlabel('sample')
ylabel('value')
legend('real', 'predicted')

subplot(212);
plot(xV_test, 'color' , 'blue')
hold on
plot(preM, 'color', 'red')
title('Stationary timeseries Prediction' , 'FontSize', 14)
xlabel('sample')
ylabel('value')
legend('real', 'predicted')

nrmse_vec(1,1) = nrmseV;
error_vec(:,1) = predicted - week_data(n-nlast+2:end);
nrmse_vec(1,2) = nrmse(week_data(n-nlast+2:end), predicted(1:end));
%saveas(gcf, ['Prediction_window_', num2str(i)], 'svg');

figure;
plot(nrmse_vec(1)*ones(1,10))
hold on
plot(nrmse_vec(2)*ones(1,10))
xlabel('window of timeseries')
ylabel('NRMSE Measure')
title('NRMSE for 1 step prediction');
legend('stationary', 'non stationary')
%saveas(gcf, ['NRMSE_window_all'], 'svg');

%% non linear
% a. attractor 2d, 3d
t = 1;
xM = embeddelays(detrended, 3, t);
plotd2d3(xM, 'Attractor', '.-')

%no need for these
figure;
autocorrelation(detrended, 20)
figure;
mutM = mutualinformation(detrended, 20, [], 'MUT', 'b');

% find m using false nearest neighbors
fnnM = falsenearest(detrended, t, 5, [], [], ' ');

% find correlation dimension for m = 1,...,5
[rcM,cM,rdM,dM,nuM] = correlationdimension(detrended,t, 6, ' ');

% prediction with different parameters
nnei = 20;
Tmax = 1;
q = 4;
for m=2:4
    for nnei=4:20
        [nrmseV,preM] = localpredictnrmse(detrended,nlast,t,m,Tmax,nnei,q,'')
        nrmse(nnei-3, 1) = nrmseV;
        [nrmseV1,preM1] = localpredictnrmse(detrended,nlast,t,m,Tmax,nnei,0,'')
        nrmse(nnei-3, 2) = nrmseV1;
        [nrmseV2,preM2] = localpredictnrmse(detrended,nlast,t,m,Tmax,nnei,2,'')
        nrmse(nnei-3, 3) = nrmseV2;
    end
    figure; 
    subplot(3,1,1)
    plot(4:20,nrmse(:,1))
    title(['Least squares for m = ', num2str(m)],'fontweight', 'bold')
    xlabel('Number of neighbors ')
    ylabel('NRMSE')

    subplot(3,1,2)
    plot(4:20,nrmse(:,2))
    title(['Local average for m = ', num2str(m)],'fontweight', 'bold')
    xlabel('Number of neighbors ')
    ylabel('NRMSE')

    subplot(3,1,3);
    plot(4:20,nrmse(:,3))
    title(['Principal Component Regression for m = ', num2str(m), ' and q = ', num2str(2)],'fontweight', 'bold')
    xlabel('Number of neighbors')
    ylabel('NRMSE')
end

    
