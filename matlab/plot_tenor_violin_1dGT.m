function plot_tenor_violin_1dGT(Results_Table, instrument)
% PLOT_TENOR_VIOLIN_CLOSEST - Violin plots for V and Rg reliability.
% Updated to show relative Rg discrepancy and filtered by Solver Winner.

if isstruct(Results_Table)
    data_tab = struct2table(Results_Table);
else
    data_tab = Results_Table;
end

% Instrument-specific noise formatting
if isstruct(instrument)
    dq = 4*pi/instrument.lambda*instrument.det_side/instrument.SD_dist/(2*round(instrument.DETpix/2)+1);
else
    dq = 0.0024;
end

% Check if True_Rg exists to switch between absolute and relative plots
hasTrueRg = ismember('True_Rg', data_tab.Properties.VariableNames);

% LaTeX Noise Formatter
latex_noise = @(x) regexprep(sprintf('$%0.1e$', x), 'e[+]{0,1}(-?)0*(\d+)', '\\times 10^{$1$2}');

% Filter out noiseless reference
data_tab = data_tab(data_tab.Noise ~= 0, :);
noislist = unique(data_tab.Noise, 'stable');
num_n = numel(noislist);
colors = lines(num_n);
colors = ones(num_n,1)*lines(1); %single color
jitter_width = 0.2;

%% --- Figure 1: V Discrepancy ---
figure(1); clf; hold on; grid on;
plot_violin_core(data_tab, noislist, 'V', dq, colors, jitter_width, latex_noise, hasTrueRg);
ylabel('$V^{1/2}$ \textrm{discrepancy}', 'Interpreter', 'latex');
% title('\textrm{Convergence and Validity of Closest } $V$ \textrm{ Solutions}', 'Interpreter', 'latex');
%% --- Figure 3: p Discrepancy ---
figure(3); clf; hold on; grid on;
plot_violin_core(data_tab, noislist, 'p', dq, colors, jitter_width, latex_noise, hasTrueRg);
ylabel('$p$ \textrm{discrepancy}', 'Interpreter', 'latex');
% title('\textrm{Convergence and Validity of Closest } $p$ \textrm{ Solutions}', 'Interpreter', 'latex');

%% --- Figure 2: Rg Plot ---
figure(2); clf; hold on; grid on;
plot_violin_core(data_tab, noislist, 'Rg', dq, colors, jitter_width, latex_noise, hasTrueRg);

if hasTrueRg
    ylabel('$\Delta R_g / R_{g,true}$ \textrm{ (Relative)}', 'Interpreter', 'latex');
%     title('\textrm{Relative Discrepancy of Extracted } $R_g$', 'Interpreter', 'latex');
    yline(0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
else
    ylabel('$R_g$ \textrm{ (nm)}', 'Interpreter', 'latex');
%     title('\textrm{Distribution of Extracted } $R_g$', 'Interpreter', 'latex');
end
end

function plot_violin_core(data_tab, noislist, mode, dq, colors, jitter_width, latex_noise, hasTrueRg)
% Core plotting logic with updated validation and Rg discrepancy math
num_n = numel(noislist);
all_means = nan(1, num_n);
all_upper = nan(1, num_n);
all_lower = nan(1, num_n);
robust_pct = 95;

for i = 1:num_n
    curr_noise = noislist(i);
    sub = data_tab(data_tab.Noise == curr_noise, :);
    
    % Define validity per user requirement
    valid_idx = ~isnan(sub.V_est);
    num_valid = sum(valid_idx);
    total_attempts = height(sub);
    p_valid = (num_valid / total_attempts) * 100;
    
    if num_valid == 0, continue; end
    
    v_all = sub.V_est(valid_idx);
    true_v = sub.True_V(valid_idx);
    
    p_all = sub.p_est(valid_idx);
    true_p = sub.True_p(valid_idx);
    
    if strcmpi(mode, 'V')
        % Calculate V Discrepancy
        plot_data = zeros(num_valid, 1);
        plot_data_p = zeros(num_valid, 1);
        for k = 1:num_valid
            plot_data(k) = sqrt(max(0, v_all(k))) - sqrt(true_v(k));
            plot_data_p(k) = true_p(k);
        end
    elseif strcmpi(mode, 'p')
        % Calculate p Discrepancy
        plot_data = zeros(num_valid, 1);
        plot_data_p = zeros(num_valid, 1);
        for k = 1:num_valid
            plot_data(k) = ( p_all(k)) - (true_p(k));
            plot_data_p(k) = true_p(k);
        end
    else
        % Extract Rg logic
        rg_all = sub.Rg_in_est(valid_idx);
        plot_data = zeros(num_valid, 1);
        plot_data_p = zeros(num_valid, 1);
        
        if hasTrueRg
            true_rg = sub.True_Rg(valid_idx);
            for k = 1:num_valid
                % Relative discrepancy: (Extracted - True) / True
                plot_data(k) = (rg_all(k) - true_rg(k)) / true_rg(k);
            plot_data_p(k) = true_p(k);
            end
        else
            for k = 1:num_valid
                [~, idx] = min(abs(v_all{k} - true_v(k)));
                plot_data(k) = rg_all{k}(idx);
            plot_data_p(k) = true_p(k);
            end
        end
    end

    % Statistics for trend lines
    avg = mean(plot_data, 'omitnan');
    r_std = std(plot_data, 'omitnan');
    all_means(i) = avg;
    all_upper(i) = avg + r_std;
    all_lower(i) = avg - r_std;
    
    % Draw Violin Patch
    [counts, centers] = histcounts(plot_data, 30);
    centers = centers(1:end-1) + diff(centers)/2;
    counts = (counts / (max(counts) + eps)) * 0.4;
    patch([i - counts, fliplr(i + counts)], [centers, fliplr(centers)], colors(i,:), ...
        'FaceAlpha', 0.25, 'EdgeColor', colors(i,:), 'HandleVisibility', 'off');
    % --- 6. VALIDITY ANNOTATION ---
    % Plotted at the top of the violin/axis
    y_pos = 0.1; % Adjust based on your Y-limits
    text(i, y_pos, sprintf('%d%%\n valid', round(p_valid)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 9, ...
        'FontWeight', 'bold' ...
        );
    
    % Jittered Points
    x_jitter = i + (rand(size(plot_data)) - 0.5) * jitter_width; %jitter
    x_jitter = i + (plot_data_p/mean(plot_data_p) - 1) * jitter_width;  %x offset prop to p
    plot(x_jitter, plot_data, '.', 'Color', [0.5 0.5 0.5], 'MarkerSize', 6, 'HandleVisibility', 'off');
    
    % Local Mean markers
    line([i-0.2, i+0.2], [avg, avg], 'Color', 'r', 'LineWidth', 2, 'HandleVisibility', 'off');
end

% Global Trend Lines
if num_n > 1
    m_label = '\textrm{Mean discrepancy}'; 
    if strcmpi(mode, 'Rg') && ~hasTrueRg, m_label = '\textrm{Mean } $R_g$'; end
    
    plot(1:num_n, all_means, 'r-', 'LineWidth', 1.2, 'DisplayName', m_label);
    
    % Properly escaped LaTeX percent for legend
    upper_label = sprintf('\\textrm{Robust (%d\\%%)} $\\pm$ 1 \\textrm{ STD}', robust_pct);
    plot(1:num_n, all_upper, 'k--', 'LineWidth', 1, 'DisplayName', upper_label);
    plot(1:num_n, all_lower, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Final Formatting
set(gca, 'TickLabelInterpreter', 'latex', 'XTick', 1:num_n);
xticklabels(arrayfun(@(x) latex_noise(abs(x)/dq^2), noislist, 'UniformOutput', false));
xlabel('\textrm{Photon density (photon/nm}$^{-2}$\textrm{)}', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
end