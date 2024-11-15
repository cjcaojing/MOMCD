function [n,adj] = network_parameters(net)

if net==1
    % net1 = Karate
    % karate %  
    n = 34; 
    c = 3;
    adj = karate();
    disp(adj)
    threshold_value = 0.5;  % clean_up_random.m
    optimal_Q=0.4198;
    Gen = 400;
    %Gen = 100;
    
elseif net==2
    % net2 = dolphin
    % dolphin %  
    n = 62; 
    c = 5;
    adj = dolphin();
    threshold_value = 0.25;  
    optimal_Q=0.5285;
    %Gen = 150; %SOSCD
    Gen = 600;
    
elseif net==3
    % net3 = polBooks
    % polBooks %  
    n = 105; 
    c = 5;
    adj = polBooks();
    threshold_value = 0.5;  
    optimal_Q=0.5272;
    Gen = 500;
    
elseif net==4
    % net4 = football
    % football %  
    n = 115; 
    c = 10;
    adj = football();
    threshold_value = 0.5;  
    optimal_Q=0.6046;
    Gen = 500;
    
elseif net==5
    % net5 = Netscience
    n = 1589; 
    el = textread('C:\Users\jing\Desktop\SOSCD\netscience_new.txt');
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1),el(i,2))=1;
        adj(el(i,2),el(i,1))=1;
    end
    threshold_value = 0.25;
    optimal_Q=0.9599;
    Gen = 100;
    
elseif net==6
    % net6 = powergrid
    n = 4941; 
    el = textread('C:\Users\jing\Desktop\SOSCD\powergrid_new.txt');
    % % min(min(el))
    % % max(max(el))
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1),el(i,2))=1;
        adj(el(i,2),el(i,1))=1;
    end
    threshold_value = 0.25;
    optimal_Q=0.9376;
    Gen = 50;
    
elseif net==7
    % net7 = 16节点模糊重叠网络
    n = 16; 
    c = 4;
    el = textread('C:\Users\jing\Desktop\SOSFCD\simple.txt');
    % % min(min(el))
    % % max(max(el))
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1),el(i,2))=1;
        adj(el(i,2),el(i,1))=1;
    end
    threshold_value = 0.25;
    optimal_Q = 0.5208;
    Gen =100;
 
 elseif net==8  % 边数有问题
    % net8 = Sawmill communication network
    % Sawmill %  
    n = 36;  % m=62 ???
    c = 4;
    el = textread('C:\Users\jing\Desktop\SOSFCD\Sawmill.txt'); 
    % % min(min(el))
    % % max(max(el))
    % % length(el)
    adj = edgeL2adj(el);
    threshold_value = 0.5;  % clean_up_random.m
    optimal_Q=0.5501;
    Gen = 100;
  
elseif net==9
    % net9 = Jazz musicians network
    % Jazz %  
    n = 198;  % m=2742  
    c = 4;
    el = textread('jazz_0.txt'); 
    % % min(min(el))
    % % max(max(el))
    % % length(el)
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end   
 elseif net==10   % 边数有问题
    % net10 = Metabolic
    % Metabolic %  
    n = 453;  % m=2025  ??? 2040!=2025
    c = 18;
    el = textread('C:\Users\jing\Desktop\SOSFCD\Metabolic_edges.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = edgeL2adj(el);
    % % sum(sum(adj))  
    threshold_value = 0.25;  % clean_up_random.m
    optimal_Q=0.4280;
    Gen = 200;
    
 elseif net==11
    % net11 = lesmis
    % lesmis %  
    n = 77;  % m=254 
    c = 5 ;
    el = textread('lesmis.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj1 = edgeL2adj(el);
    adj=adj1+adj1'
    disp(adj)
    % % sum(sum(adj))  
    threshold_value = 0.25;  % clean_up_random.m
    optimal_Q=0.5687;
    Gen = 700;
    
 elseif net==12
    % net12 = words
    % words %  
    n = 112;  % m=425 
    c = 7 ;
    el = textread('C:\Users\jing\Desktop\SOSFCD\words.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = edgeL2adj(el);
    % % sum(sum(adj))  
    threshold_value = 0.25;  % clean_up_random.m
    optimal_Q=0.7159;
    Gen = 150;
    
 elseif net==13       % 边数有问题
    % net13 = email
    n = 1133;  % m=5451 != 10903
    c = 9 ;
    el = textread('C:\Users\jing\Desktop\SOSFCD\email.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = edgeL2adj(el(:,1:2));
    % % sum(sum(adj))  
    threshold_value = 0.25;  % clean_up_random.m
    optimal_Q=0.5608;
    Gen = 150;

elseif net==14       % zhang_network
    n =13;
    c=3;
    el = textread('C:\Users\jing\Desktop\SOSFCD\zhang_network.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1),el(i,2))=1;
        adj(el(i,2),el(i,1))=1;
    end
   % find(adj(13,:))
    threshold_value = 0.25;  % clean_up_random.m
    optimal_Q=0.3699;
    Gen = 100;

elseif net==15       % netscience_maxconnected 319/914
    n =379;
    c=20;
    el = textread('C:\Users\jing\Desktop\SOSFCD\netscience_maxconnected.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)=914
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1),el(i,2))=1;
        adj(el(i,2),el(i,1))=1;
    end
   % find(adj(13,:))
    threshold_value = 0.5;  % clean_up_random.m
    optimal_Q=0.8506;
    Gen = 200;
elseif net==16       % network_brain_47
    n =47;
    el = textread('network_brain_47.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)=914
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
elseif net==17       % road-minnesota_0
    n =2642;
    el = textread('road-minnesota_0.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)=914
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
elseif net==18       % metabolic_0
    n =453;
    el = textread('metabolic_0.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)=914
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==19       
    % net19 = email
    n = 1133;  % m=5451
    c = 9 ;
    el = textread('email_0.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==20       
    % netscience
    n = 1461;  
    c = 9 ;
    el = textread('netscience_lianxu_wu.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==21       
    % facebook
    n = 1589;  % m=5451
    c = 9 ;
    el = textread('facebook_2888_0.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==22       
    % powergrid
    n = 4941;  % m=5451
    c = 9 ;
    el = textread('powergrid_0.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==23       
    % facebook
    n = 10680;  % m=5451
    c = 9 ;
    el = textread('PGP_0.txt'); 
    % % min(min(el(:,1:2)))
    % % max(max(el(:,1:2)))
    % % length(el)
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==24     
    % network0
    n = 1000; 
    el = textread('network00_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==25      
    % network1
    n = 1000; 
    el = textread('network10_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==26      
    % network2
    n = 1000; 
    el = textread('network20_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==27    
    % network3
    n = 1000; 
    el = textread('network30_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==28     
    % network4
    n = 1000; 
    el = textread('network40_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==29     
    % network5
    n = 1000; 
    el = textread('network50_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==30     
    % network6
    n = 1000; 
    el = textread('network60_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==31   
    % network7
    n = 1000; 
    el = textread('network70_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==32   
    % network8
    n = 1000; 
    el = textread('network80_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==33     
    % network9
    n = 1000; 
    el = textread('network90_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==34     
    % gn
    n = 128; 
    el = textread('GN01.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==35     
    % gn
    n = 128; 
    el = textread('GN11.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==36     
    % gn
    n = 128; 
    el = textread('GN21.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==37    
    % gn
    n = 128; 
    el = textread('GN31.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==38    
    % gn
    n = 128; 
    el = textread('GN41.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==39   
    % gn
    n = 128; 
    el = textread('GN51.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==40   
    % gn
    n = 128; 
    el = textread('GN61.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==41   
    % gn
    n = 128; 
    el = textread('GN71.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==42  
    % gn
    n = 128; 
    el = textread('GN81.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==43  
    % gn
    n = 128; 
    el = textread('GN91.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==44 
    % gn
    n = 128; 
    el = textread('GN02.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==45 
    % cornell
    n = 195; 
    el = textread('cornell_0.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
 elseif net==46 
    % cora_edges
    n = 2708; 
    el = textread('cora_edges.txt'); 
    adj = zeros(n,n);
    for i=1:length(el)
        adj(el(i,1)+1,el(i,2)+1)=1;
        adj(el(i,2)+1,el(i,1)+1)=1;
    end
end

