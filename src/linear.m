%% Time Series Project 2018 - 2019
% Aristotle University of Thessaloniki, 
% Dept. of Electrical and Computer Engineering 
%
% Authors : Athanasiadis Christos, 8416
%           Matsoukas Vasileios, 8743 
%

close all
clear 

%% Study A
data = importdata('dat4.dat');
window = 250;
K = floor(length(data)/window);
% data = data(1 : K*window) ; 

win_data = reshape(data(1 : K*window),window, K);

data = data(1 : K*window + 1) ;
diffs = diff(data, 1);
diffs = reshape(diffs,window, K);


alpha = 0.05;
maxtau = 20;
zalpha = norminv(1-alpha/2);
autlim = zalpha/sqrt(window);
important_autocorrs = zeros(1, K);


Tmax = 10;
max_p = 5;
max_q = 5;
aic_fit = zeros(max_p+1, max_q+1);
nrmse_fit = cell(max_p+1, max_q+1);

for i = 1:K
    
    % Visual representation of data
    figure('name', ['Data_window_', ' ', num2str(i)]);
    g = suptitle({['Window', ' ', num2str(i)],' ',' '});
    set(g, 'FontSize', 12, 'FontWeight', 'bold')
    subplot(231);
    plot(win_data(:, i))
    ylabel('Exchange rate'); % DEN EIMAI KAI POLU SIGOUROS EDW.?!
    xlabel('Sample')
    title('Original Time Series')
    
    % Trend removal using first differences on time series      
    subplot(232);
    plot(diffs(:, i))
    ylabel('Exchange rate'); % DEN EIMAI KAI POLU SIGOUROS EDW.?!
    xlabel('Sample')
    title('Detrended Time Series (First Diffs)'); 
    
    % Stationary time series autocorellation & partial autocorellation 
    acM = autocorrelation(diffs(:, i), maxtau);
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
    important_autocorrs(i) = length(find(abs(acM(:,2))>autlim)) - 1; %sub 1 because at lag 0, it's always 1
    hold off
    
    pacfV = parautocor(diffs(:, i), maxtau);
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
    acM = autocorrelation(diffs(:, i), maxtau);
    tittxt = sprintf('Ljung-Box Test');
    subplot(235)
    [h2V,p2V,Q2V] = portmanteauLB(acM(2:maxtau+1,2),window,alpha);
    plot([1:maxtau]',p2V,'.-k')
    hold on
    plot([0 maxtau+1],alpha*[1 1],'--c')   
    xlabel('lag \tau')
    ylabel('p-value')
    title(' Ljung-Box Portmanteau test')
    axis([0 maxtau+1 0 1])
    
    % Fit of an ARMA model
    
   
    if(sum(h2V) > 0)
    
        for p = 0:max_p
            for q = 0:max_q
                    [nrmseV,phiallV,thetaallV,SDz,aicS,fpeS] = fitARMA(diffs(:, i),p,q,Tmax);
                    aic_fit(p+1,q+1) = aicS; 
                    nrmse_fit{p+1,q+1} = nrmseV;
            end
        end

         min_aic = min(min(aic_fit));
        [min_aic_p, min_aic_q] = find(aic_fit == min_aic);
        
        fprintf('Window %d :\n',i);
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
        fprintf('\n Window %d : White Noise Time Series -> NRMSE = 1. \n',i);
        
    end
    
%     saveas(gcf, ['Time_Series_Window_', num2str(i)], 'svg');
end


% Plot histogram with logarithm differences detrending method
% for i = 1 : K
%     figure;
%     hist(log_diffs(i,:), 20)
%     %saveas(gcf, ['Histogram_window_', num2str(i)], 'svg');
% end


%% Prediction (1 step)

p = 5;
q = 0;
Tmax = 1 ;
nrmse_vec = zeros(K,2);
error_vec = zeros(100-1,K);

for i = 1:K
    
    n = length(diffs(:, i)); %length(data((i - 1) * window + 1 : i * window)); % time series length   
    proptest = 0.4; % proportion of the whole time series to be used as test 100/250

    xV = diffs(:, i); %data((i - 1) * window + 1 : i * window);
    xV_train = xV(1:(1 - proptest)*length(xV)); % training set
    xV_test = xV((1 - proptest)*length(xV)+1:end); % testing set
    nlast = proptest*n;
    
    [nrmseV,preM,phiV,thetaV] = predictARMAnrmse(xV,p,q,Tmax,nlast);
    

    figure('name', ['Data_window_', ' ', num2str(i)]);
    
    subplot(211);
    plot(xV_test, 'color' , 'blue')
    hold on
    plot(preM, 'color', 'red')
    title(['Stationary timeseries Prediction for window ', num2str(i)], 'FontSize', 14)
    xlabel('sample')
    ylabel('value')
    legend('real', 'predicted')
    
%     predicted = preM + win_data(n-nlast:end-1,i);
    
    predicted = preM(1:end-1) + win_data(n-nlast+1:end-1,i);
   
    subplot(212);
    plot(win_data(n-nlast+2:end, i), 'color' , 'blue')
%     subplot(212);
%     plot(win_data(n-nlast+1:end, i), 'color' , 'blue')
    hold on
    plot(predicted, 'color', 'red')
    title(['Prediction for window ', num2str(i)], 'FontSize', 14)
    xlabel('sample')
    ylabel('value')
    legend('real', 'predicted')
    nrmse_vec(i,1) = nrmseV;
    error_vec(:,i) = predicted - win_data(n-nlast+2:end, i);
    nrmse_vec(i,2) = nrmse(win_data(n-nlast+2:end, i), predicted(1:end));
%     error_vec(:,i) = predicted - win_data(n-nlast+1:end, i);
%     nrmse_vec(i,2) = nrmse(win_data(n-nlast+1:end, i), predicted(1:end));
    %saveas(gcf, ['Prediction_window_', num2str(i)], 'svg');

end

figure;
plot(nrmse_vec)
xlabel('window of timeseries')
ylabel('NRMSE Measure')
title('NRMSE for 1 step prediction');
legend('stationary', 'non stationary')

%saveas(gcf, ['NRMSE_window_all'], 'svg');
