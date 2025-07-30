/**
 * 交互式人体姿态调节器 - JavaScript模块
 * 包含所有前端交互逻辑和播放功能
 */

class PoseAdjuster {
    constructor() {
        this.currentFrame = 0;
        this.frameCount = 5;
        this.currentAngles = {}; // 存储当前帧的绝对角度值
        this.refreshTimeout = null; // 延迟刷新的定时器
        this.pendingChanges = {}; // 存储累积的角度变化
        this.jointConfigs = null; // 从JSON文件加载的关节配置
        this.modifiedJoints = new Set(); // 跟踪已修改的关节轴
        this.originalParams = {}; // 存储原始参数值
        this.init();
    }

    async init() {
        // 首先加载关节配置
        await this.loadJointConfigs();
        
        this.initControls();
        this.initEventListeners();
        this.loadSequenceConfig();

        // 立即进行一次初始位置调整
        this.adjustControlPanelPosition();

        this.render();
        this.updateDataInfo();

        // 延迟调整位置，确保DOM完全渲染
        setTimeout(() => {
            this.forceAdjustPosition(); // 使用强制调整
            this.checkAndAdjustFrameNumbers(); // 检查并调整数字按钮
        }, 100);

        // 额外的延迟检查，确保页面完全加载后调整
        setTimeout(() => {
            this.checkAndAdjustFrameNumbers();
        }, 500);

        // 最后的保障检查
        setTimeout(() => {
            this.checkAndAdjustFrameNumbers();
        }, 1000);

        // 监听窗口大小变化，重新调整位置
        // 窗口大小变化时调整面板位置和数字按钮宽度
        window.addEventListener('resize', () => {
            this.adjustControlPanelPosition();
            this.adjustFrameNumbersWidth();
        });

        // 监听渲染区域内容变化，实时调整位置
        const renderArea = document.querySelector('.render-area');
        if (renderArea) {
            // 使用MutationObserver监听DOM变化
            const observer = new MutationObserver(() => {
                setTimeout(() => this.adjustControlPanelPosition(), 50);
            });
            observer.observe(renderArea, {
                childList: true,
                subtree: true,
                attributes: true
            });
        }

        // 监听图像加载完成，重新调整位置
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => this.forceAdjustPosition(), 200);
        });
    }

    /**
     * 从JSON文件加载关节配置
     */
    async loadJointConfigs() {
        try {
            const response = await fetch('/static/joint_config.json');
            const data = await response.json();
            this.jointConfigs = data.core_joints;
            console.log('关节配置加载成功:', Object.keys(this.jointConfigs));
        } catch (error) {
            console.error('关节配置加载失败:', error);
            throw new Error('无法加载关节配置，请检查 joint_config.json 文件');
        }
    }

    adjustControlPanelPosition() {
        // 动态调整控制面板位置，基于预览区域的实际高度
        const renderArea = document.querySelector('.render-area');
        const controlPanel = document.querySelector('.control-panel');

        if (renderArea && controlPanel) {
            // 获取渲染区域的实际高度
            const renderAreaRect = renderArea.getBoundingClientRect();
            const renderAreaHeight = renderAreaRect.height;

            // 确保有足够间距，但不设置最小值限制
            const marginTop = renderAreaHeight + 30;
            controlPanel.style.marginTop = `${marginTop}px`;
        }
    }

    forceAdjustPosition() {
        // 强制多次调整位置，解决首次加载问题
        this.adjustControlPanelPosition();
        setTimeout(() => this.adjustControlPanelPosition(), 100);
        setTimeout(() => this.adjustControlPanelPosition(), 300);
        setTimeout(() => this.adjustControlPanelPosition(), 600);
        setTimeout(() => this.adjustControlPanelPosition(), 1000);
    }

    async updateDataInfo() {
        // 从服务器获取数据信息
        try {
            const response = await fetch('/api/data_info');
            const data = await response.json();
            if (data.success) {
                document.getElementById('dataInfo').textContent =
                    `帧率: ${data.framerate}Hz | 总帧数: ${data.total_frames}`;
            }
        } catch (error) {
            // 静默失败，使用默认值
        }
    }

    async loadSequenceConfig() {
        // 从服务器获取序列配置参数并设置到页面
        try {
            const response = await fetch('/api/sequence_config');
            const data = await response.json();
            if (data.success) {
                document.getElementById('startFrameInput').value = data.start_frame;
                document.getElementById('frameIntervalInput').value = data.frame_interval;
                document.getElementById('numFramesInput').value = data.num_frames;
                document.getElementById('frameOffsetInput').value = data.frame_offset;
                
                // 更新原始参数值（用于状态检测）
                this.originalParams['startFrameInput'] = data.start_frame.toString();
                this.originalParams['frameIntervalInput'] = data.frame_interval.toString();
                this.originalParams['numFramesInput'] = data.num_frames.toString();
                this.originalParams['frameOffsetInput'] = data.frame_offset.toString();
            }
        } catch (error) {
            // 静默失败，使用空值
        }
    }

    async loadCurrentAngles() {
        // 获取当前帧的关节绝对角度并更新UI显示
        this.updateStatusInfo('loading', '正在获取角度数据...');

        try {
            const response = await fetch('/api/current_angles');
            const data = await response.json();
            if (data.success) {
                this.currentAngles = data.joint_angles;
                this.updateAllSlidersWithAngles();
                this.updateStatusInfo('success', '角度数据已更新');
            } else {
                this.currentAngles = {};
                this.updateStatusInfo('error', `错误: ${data.error}`);
            }
        } catch (error) {
            this.currentAngles = {};
            this.updateStatusInfo('error', '网络错误，无法获取角度数据');
        }
    }

    updateStatusInfo(status, message) {
        const statusFloat = document.getElementById('statusFloat');
        if (statusFloat) {
            const statusDot = statusFloat.querySelector('.status-dot');
            const statusText = statusFloat.querySelector('.status-text');

            if (statusDot) {
                statusDot.className = `status-dot ${status}`;
            }
            if (statusText) {
                statusText.textContent = message;
            }

            // 显示状态浮窗
            statusFloat.className = `status-float show ${status}`;

            // 3秒后自动隐藏（成功状态）
            if (status === 'success') {
                setTimeout(() => {
                    statusFloat.className = 'status-float';
                }, 3000);
            }
        }
    }

    updateAllSlidersWithAngles() {
        // 更新所有滑块显示当前绝对角度值
        Object.entries(this.currentAngles || {}).forEach(([jointName, axes]) => {
            Object.entries(axes).forEach(([axisIdx, currentAngle]) => {
                const axisIndex = parseInt(axisIdx);
                this.updateSliderValue(jointName, axisIndex, currentAngle);
            });
        });
    }

    initControls() {
        // 使用动态加载的关节配置
        if (!this.jointConfigs) {
            console.error('关节配置未加载，无法初始化控制面板');
            return;
        }
        const joints = this.jointConfigs;

        // 按组分类关节
        const groupedJoints = {};
        Object.entries(joints).forEach(([jointName, config]) => {
            const groupName = config.group;
            if (!groupedJoints[groupName]) {
                groupedJoints[groupName] = [];
            }
            groupedJoints[groupName].push([jointName, config]);
        });

        const container = document.getElementById('jointControls');

        // 为每个组创建控制面板
        Object.entries(groupedJoints).forEach(([groupName, groupJoints]) => {
            const groupDiv = document.createElement('div');
            groupDiv.className = 'joint-group';

            const groupHeader = document.createElement('div');
            groupHeader.className = 'group-header';
            groupHeader.textContent = groupName;
            groupDiv.appendChild(groupHeader);

            groupJoints.forEach(([jointName, config]) => {
                const jointDiv = document.createElement('div');
                jointDiv.style.marginBottom = '12px';

                const titleDiv = document.createElement('div');
                titleDiv.className = 'joint-title';
                titleDiv.textContent = config.name;
                jointDiv.appendChild(titleDiv);

                config.axes.forEach((axisName, axisIdx) => {
                    const axisDiv = document.createElement('div');
                    axisDiv.className = 'axis-control';

                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'axis-label';
                    labelDiv.textContent = axisName; // 只显示轴名称，不显示范围

                    const controlRow = document.createElement('div');
                    controlRow.className = 'control-row';

                    // 增量输入框
                    const incrementInput = document.createElement('input');
                    incrementInput.type = 'number';
                    incrementInput.className = 'increment-input';
                    incrementInput.value = 5;
                    incrementInput.min = 1;
                    incrementInput.max = 45;
                    incrementInput.id = `increment_${jointName}_${axisIdx}`;
                    
                    // 为增量输入框添加滚轮支持，但阻止页面滚动
                    incrementInput.addEventListener('wheel', (e) => {
                        e.preventDefault(); // 阻止页面滚动
                        e.stopPropagation(); // 阻止事件冒泡
                        
                        const currentValue = parseFloat(e.target.value) || 5;
                        const increment = e.deltaY > 0 ? -1 : 1; // 向下滚动减少，向上滚动增加
                        const newValue = Math.max(1, Math.min(45, currentValue + increment)); // 限制在1-45范围内
                        
                        e.target.value = newValue;
                        // 触发change事件以更新step值
                        e.target.dispatchEvent(new Event('change'));
                    });

                    const minusBtn = document.createElement('button');
                    minusBtn.className = 'micro-btn';
                    minusBtn.textContent = '−';
                    minusBtn.onclick = () => {
                        const increment = parseFloat(incrementInput.value) || 5;
                        this.adjustAngleWithDelay(jointName, axisIdx, -increment, 'add');
                    };

                    const plusBtn = document.createElement('button');
                    plusBtn.className = 'micro-btn';
                    plusBtn.textContent = '+';
                    plusBtn.onclick = () => {
                        const increment = parseFloat(incrementInput.value) || 5;
                        this.adjustAngleWithDelay(jointName, axisIdx, increment, 'add');
                    };

                    const resetBtn = document.createElement('button');
                    resetBtn.className = 'reset-joint-btn';
                    resetBtn.textContent = '↺';
                    resetBtn.title = '重置此轴';
                    resetBtn.id = `reset_${jointName}_${axisIdx}`;
                    resetBtn.disabled = true; // 初始状态为禁用
                    resetBtn.style.opacity = '0.5'; // 视觉上显示为禁用状态
                    resetBtn.onclick = () => this.resetJoint(jointName, axisIdx);

                    const valueInput = document.createElement('input');
                    valueInput.type = 'number';
                    valueInput.className = 'angle-input';
                    valueInput.id = `input_${jointName}_${axisIdx}`;
                    valueInput.value = '0.00';
                    valueInput.min = -200;
                    valueInput.max = 200;
                    valueInput.step = 5; // 与增量输入框默认值一致
                    valueInput.title = `关节调节范围: ${config.limits[axisIdx][0]}° ~ ${config.limits[axisIdx][1]}°`;
                    
                    // 存储原始值，用于验证失败时恢复
                    valueInput.onfocus = (e) => {
                        e.target.dataset.originalValue = e.target.value;
                    };
                    
                    const validateAndSet = (e) => {
                        const inputValue = parseFloat(e.target.value);
                        const originalValue = parseFloat(e.target.dataset.originalValue || '0');
                        
                        // 检查值是否真的发生了变化
                        if (Math.abs(inputValue - originalValue) < 0.01) {
                            // 值没有实质性变化，不需要刷新
                            return;
                        }
                        
                        if (isNaN(inputValue)) {
                            this.showWarning('请输入有效的数字');
                            e.target.value = e.target.dataset.originalValue || '0.00';
                            return;
                        }
                        
                        // 检查范围
                        const limits = config.limits[axisIdx];
                        const [minVal, maxVal] = limits;
                        
                        if (inputValue < minVal || inputValue > maxVal) {
                            this.showWarning(`角度超出范围 (${minVal}° ~ ${maxVal}°)`);
                            e.target.value = e.target.dataset.originalValue || '0.00';
                            return;
                        }
                        
                        // 值有效，进行调整
                        this.adjustAngleWithDelay(jointName, axisIdx, inputValue, 'set');
                    };
                    
                    valueInput.onchange = validateAndSet;
                    valueInput.onblur = validateAndSet;
                    valueInput.onkeypress = (e) => {
                        if (e.key === 'Enter') {
                            e.target.blur();
                        }
                    };
                    
                    // 确保角度输入框即使聚焦时也不响应滚轮
                    valueInput.addEventListener('wheel', (e) => {
                        e.preventDefault(); // 阻止数值调整
                        // 不调用stopPropagation()，让事件冒泡到父元素，允许页面滚动
                    });
                    


                    // 动态更新step值，与增量输入框保持同步
                    incrementInput.onchange = () => {
                        const incrementValue = parseFloat(incrementInput.value) || 5;
                        valueInput.step = incrementValue;
                    };

                    const degreeLabel = document.createElement('span');
                    degreeLabel.className = 'degree-label';
                    degreeLabel.textContent = '°';

                    // 范围提示放在角度输入框后面
                    const rangeHint = document.createElement('span');
                    rangeHint.className = 'range-hint';
                    const range = config.limits[axisIdx];
                    rangeHint.textContent = `(${range[0]}° ~ ${range[1]}°)`;

                    controlRow.appendChild(minusBtn);
                    controlRow.appendChild(incrementInput);
                    controlRow.appendChild(plusBtn);
                    controlRow.appendChild(resetBtn);
                    controlRow.appendChild(valueInput);
                    controlRow.appendChild(degreeLabel);
                    controlRow.appendChild(rangeHint);

                    // 将标签和控件放在同一行
                    axisDiv.appendChild(labelDiv);
                    axisDiv.appendChild(controlRow);
                    jointDiv.appendChild(axisDiv);
                });

                groupDiv.appendChild(jointDiv);
            });

            container.appendChild(groupDiv);
        });
    }

    initEventListeners() {
        // 角度调节控制
        document.addEventListener('input', (e) => {
            if (e.target.matches('.angle-input, .increment-input')) {
                this.handleAngleChange(e);
            }
        });

        // 底部功能按钮事件绑定
        this.initBottomButtons();

        // 添加输入框实时验证
        this.addInputValidation();
    }

    /**
     * 初始化底部功能按钮的事件监听器
     */
    initBottomButtons() {
        // 复制上一帧按钮
        const copyPrevBtn = document.getElementById('copyPrevBtn');
        if (copyPrevBtn) {
            copyPrevBtn.addEventListener('click', () => this.copyPrevFrame());
        }

        // 重置当前帧按钮
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetCurrentFrame());
        }

        // 导出序列图像按钮
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportSequenceImages());
        }
    }

    /**
     * 复制上一帧的调节参数
     */
    async copyPrevFrame() {
        if (this.currentFrame === 0) {
            this.updateStatusInfo('error', '当前已是第一帧，无法复制上一帧参数');
            return;
        }

        this.setLoading(true);
        this.updateStatusInfo('loading', '正在复制上一帧参数...');

        try {
            const response = await fetch('/api/copy_prev', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            if (result.success) {
                // 重新渲染并更新角度显示
                document.getElementById('renderImage').src = result.image;
                this.updateFrameStatus(result.modified);
                await this.loadCurrentAngles();
                this.updateStatusInfo('success', '复制上一帧参数成功');
            } else {
                this.updateStatusInfo('error', '复制失败: ' + result.error);
            }
        } catch (error) {
            console.error('复制上一帧失败:', error);
            this.updateStatusInfo('error', '复制失败: ' + error.message);
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * 重置当前帧的所有调节
     */
    async resetCurrentFrame() {
        if (!confirm('确定要重置当前帧的所有调节吗？此操作不可撤销。')) {
            return;
        }

        this.setLoading(true);
        this.updateStatusInfo('loading', '正在重置当前帧...');

        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            if (result.success) {
                // 重新渲染并更新角度显示
                document.getElementById('renderImage').src = result.image;
                this.updateFrameStatus(result.modified);
                await this.loadCurrentAngles();
                // 清除修改状态
                this.modifiedJoints.clear();
                this.updateAllResetButtonStates(false);
                this.updateStatusInfo('success', '重置当前帧成功');
            } else {
                this.updateStatusInfo('error', '重置失败: ' + result.error);
            }
        } catch (error) {
            console.error('重置当前帧失败:', error);
            this.updateStatusInfo('error', '重置失败: ' + error.message);
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * 导出序列图像
     */
    async exportSequenceImages() {
        this.setLoading(true);
        this.updateStatusInfo('loading', '正在导出序列图像...');

        try {
            const response = await fetch('/api/export');
            
            if (!response.ok) {
                throw new Error('导出请求失败');
            }

            const result = await response.json();
            if (result.success) {
                // 创建下载链接
                const link = document.createElement('a');
                link.href = result.image;
                link.download = result.filename || 'pose_sequence.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                this.updateStatusInfo('success', '序列图像导出成功');
            } else {
                this.updateStatusInfo('error', '导出失败: ' + (result.error || '未知错误'));
            }
        } catch (error) {
            console.error('导出序列图像失败:', error);
            this.updateStatusInfo('error', '导出失败: ' + error.message);
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * 添加输入框实时验证，防止用户输入超出范围的值
     */
    addInputValidation() {
        const inputConfigs = [
            { id: 'startFrameInput', min: 0, max: 500, name: '起始帧' },
            { id: 'frameIntervalInput', min: 1, max: 50, name: '帧间隔' },
            { id: 'numFramesInput', min: 1, max: 20, name: '帧数' },
            { id: 'frameOffsetInput', min: -100, max: 100, name: '帧偏移' }
        ];

        inputConfigs.forEach(config => {
            const input = document.getElementById(config.id);
            if (!input) return;

            // 防止鼠标滚轮事件冒泡到页面，避免页面滚动
            input.addEventListener('wheel', (e) => {
                e.stopPropagation(); // 阻止事件冒泡，但不阻止默认行为
            });

            // 实时验证输入值
            input.addEventListener('input', (e) => {
                let value = parseInt(e.target.value);
                
                if (isNaN(value)) return;
                
                if (value < config.min) {
                    e.target.value = config.min;
                    this.showInputWarning(e.target, `${config.name}最小值为${config.min}`);
                } else if (value > config.max) {
                    e.target.value = config.max;
                    this.showInputWarning(e.target, `${config.name}最大值为${config.max}`);
                }
                
                // 更新应用按钮状态
                this.updateApplyButtonState();
            });

            // 失去焦点时再次验证
            input.addEventListener('blur', (e) => {
                let value = parseInt(e.target.value);
                if (isNaN(value) || value < config.min) {
                    e.target.value = config.min;
                } else if (value > config.max) {
                    e.target.value = config.max;
                }
            });
        });
    }

    /**
     * 显示输入框警告提示
     */
    showInputWarning(inputElement, message) {
        // 移除已存在的警告
        const existingWarning = inputElement.parentNode.querySelector('.input-warning');
        if (existingWarning) {
            existingWarning.remove();
        }

        // 创建新的警告提示
        const warning = document.createElement('div');
        warning.className = 'input-warning';
        warning.textContent = message;
        
        inputElement.parentNode.appendChild(warning);
        
        // 3秒后自动移除
        setTimeout(() => {
            if (warning.parentNode) {
                warning.remove();
            }
        }, 3000);
    }

    /**
     * 为参数输入框添加滚轮支持和状态管理
     */
    initParamInputWheelSupport() {
        const paramInputs = [
            'startFrameInput',
            'frameIntervalInput', 
            'numFramesInput',
            'frameOffsetInput'
        ];

        paramInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                // 记录原始值
                this.originalParams[inputId] = input.value;

                // 滚轮支持
                input.addEventListener('wheel', (e) => {
                    e.preventDefault(); // 阻止页面滚动
                    e.stopPropagation(); // 阻止事件冒泡
                    
                    const currentValue = parseFloat(e.target.value) || 0;
                    const increment = e.deltaY > 0 ? -1 : 1; // 向下滚动减少，向上滚动增加
                    let newValue = currentValue + increment;
                    
                    // 根据不同输入框设置最小值
                    let min;
                    if (inputId === 'startFrameInput' || inputId === 'frameOffsetInput') {
                        min = 0; // 起始帧和帧偏移最小值为0
                    } else {
                        min = 1; // 帧间隔和帧数最小值为1
                    }
                    const max = parseFloat(e.target.max) || 999999;
                    
                    newValue = Math.max(min, Math.min(max, newValue));
                    e.target.value = newValue;
                    
                    // 检查是否有变化并更新应用按钮状态
                    this.updateApplyButtonState();
                });

                // 输入变化监听
                input.addEventListener('input', () => {
                    this.updateApplyButtonState();
                });

                // Enter键监听
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        const applyBtn = document.querySelector('.config-btn');
                        if (applyBtn && !applyBtn.disabled) {
                            updateSequence(); // 调用全局的应用函数
                        }
                    }
                });
            }
        });

        // 初始化应用按钮状态
        this.updateApplyButtonState();
    }

    /**
     * 更新应用和重置按钮状态
     */
    updateApplyButtonState() {
        const applyBtn = document.querySelector('.config-btn');
        const resetBtn = document.getElementById('resetParamsBtn');
        
        const paramInputs = [
            'startFrameInput',
            'frameIntervalInput', 
            'numFramesInput',
            'frameOffsetInput'
        ];

        // 检查是否有任何参数发生变化
        let hasChanges = false;
        paramInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input && input.value !== this.originalParams[inputId]) {
                hasChanges = true;
            }
        });

        // 更新应用按钮状态
        if (applyBtn) {
            applyBtn.disabled = !hasChanges;
        }
        
        // 更新重置按钮状态
        if (resetBtn) {
            resetBtn.disabled = !hasChanges;
        }
    }

    /**
     * 重置原始参数值（在成功应用后调用）
     */
    resetOriginalParams() {
        const paramInputs = [
            'startFrameInput',
            'frameIntervalInput', 
            'numFramesInput',
            'frameOffsetInput'
        ];

        paramInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                this.originalParams[inputId] = input.value;
            }
        });

        this.updateApplyButtonState();
    }

    /**
     * 立即更新显示值并验证有效性，然后延迟刷新帧
     * 支持连续点击累积
     */
    adjustAngleWithDelay(jointName, axis, angle, operation = 'add') {
        // 生成关节轴的唯一键
        const key = `${jointName}_${axis}`;
        
        // 如果这是一个新的累积序列，记录原始值
        if (!this.pendingChanges[key]) {
            const originalValue = this.getCurrentAngleValue(jointName, axis);
            this.pendingChanges[key] = {
                jointName: jointName,
                axis: axis,
                originalValue: originalValue,
                totalChange: 0,
                operation: operation
            };
        }
        
        // 立即更新显示值
        const newValue = this.updateDisplayValueImmediate(jointName, axis, angle, operation);
        
        if (newValue !== null) {
            // 更新累积变化量
            if (operation === 'add') {
                this.pendingChanges[key].totalChange += angle;
            } else {
                // 'set'操作：计算相对于原始值的变化
                this.pendingChanges[key].totalChange = newValue - this.pendingChanges[key].originalValue;
                this.pendingChanges[key].operation = 'add'; // 转换为add操作发送到服务器
            }
            
            // 清除之前的延迟刷新
            if (this.refreshTimeout) {
                clearTimeout(this.refreshTimeout);
            }
            
            // 延迟400ms后批量发送所有累积的变化
            this.refreshTimeout = setTimeout(() => {
                this.flushPendingChanges();
            }, 400);
        }
    }

    /**
     * 获取关节轴的当前角度值
     */
    getCurrentAngleValue(jointName, axis) {
        if (this.currentAngles[jointName] && this.currentAngles[jointName][axis] !== undefined) {
            return this.currentAngles[jointName][axis];
        }
        return 0;
    }

    /**
     * 批量发送所有累积的角度变化
     */
    async flushPendingChanges() {
        if (Object.keys(this.pendingChanges).length === 0) {
            return;
        }

        this.setLoading(true);

        try {
            // 构建批量调整请求
            const changes = Object.values(this.pendingChanges).map(change => ({
                joint_name: change.jointName,
                axis: change.axis,
                angle: change.totalChange,
                operation: 'add' // 所有累积变化都作为add操作发送
            }));
            
            const response = await fetch('/api/adjust_batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    changes: changes
                })
            });

            const data = await response.json();
            if (data.success) {
                document.getElementById('renderImage').src = data.image;
                // 重新获取最新的角度值
                await this.loadCurrentAngles();
                this.updateFrameStatus(true);
            } else {
                // 如果批量调整不支持，逐个发送
                for (const change of changes) {
                    await this.adjustAngle(change.joint_name, change.axis, change.angle, change.operation);
                }
            }
        } catch (error) {
            console.error('批量调节失败，尝试逐个调节:', error);
            // fallback：逐个发送变化
            const changes = Object.values(this.pendingChanges);
            for (const change of changes) {
                try {
                    await this.adjustAngle(change.jointName, change.axis, change.totalChange, 'add');
                } catch (e) {
                    console.error('单个调节失败:', e);
                }
            }
        } finally {
            // 清空累积的变化
            this.pendingChanges = {};
            this.setLoading(false);
        }
    }

    /**
     * 立即更新显示值并验证有效性
     * @param {string} jointName 关节名称
     * @param {number} axis 轴索引
     * @param {number} angle 角度值
     * @param {string} operation 操作类型 ('add' 或 'set')
     * @returns {number|null} 新的角度值，如果无效则返回null
     */
    updateDisplayValueImmediate(jointName, axis, angle, operation) {
        const jointConfig = this.getJointConfig(jointName);
        if (!jointConfig) {
            this.showWarning('无效的关节名称');
            return null;
        }

        const limits = jointConfig.limits[axis];
        if (!limits) {
            this.showWarning('无效的轴索引');
            return null;
        }

        // 获取当前显示的值（从UI中读取，保证累积显示的正确性）
        let currentValue = 0;
        const valueInput = document.getElementById(`input_${jointName}_${axis}`);
        if (valueInput && valueInput.value) {
            currentValue = parseFloat(valueInput.value) || 0;
        } else if (this.currentAngles[jointName] && this.currentAngles[jointName][axis] !== undefined) {
            currentValue = this.currentAngles[jointName][axis];
        }

        // 计算新值
        let newValue;
        if (operation === 'add') {
            newValue = currentValue + angle;
        } else {
            newValue = angle;
        }

        // 验证范围
        const [minVal, maxVal] = limits;
        if (newValue < minVal || newValue > maxVal) {
            this.showWarning(`角度超出范围 (${minVal}° ~ ${maxVal}°)`);
            return null;
        }

        // 立即更新显示
        this.updateSliderValue(jointName, axis, newValue);
        
        // 更新内存中的角度值（用于累积计算）
        if (!this.currentAngles[jointName]) {
            this.currentAngles[jointName] = {};
        }
        this.currentAngles[jointName][axis] = newValue;

        // 标记为已修改并更新重置按钮状态
        const jointKey = `${jointName}_${axis}`;
        this.modifiedJoints.add(jointKey);
        this.updateResetButtonState(jointName, axis, true);

        return newValue;
    }

    /**
     * 更新重置按钮状态
     */
    updateResetButtonState(jointName, axis, isModified) {
        const resetBtn = document.getElementById(`reset_${jointName}_${axis}`);
        if (resetBtn) {
            resetBtn.disabled = !isModified;
            resetBtn.style.opacity = isModified ? '1' : '0.5';
            resetBtn.title = isModified ? '重置此轴' : '此轴未修改';
        }
    }

    /**
     * 获取关节配置信息
     */
    getJointConfig(jointName) {
        if (!this.jointConfigs) {
            console.error('关节配置未加载');
            return null;
        }
        return this.jointConfigs[jointName];
    }

    /**
     * 显示警告提示
     */
    showWarning(message) {
        // 创建警告提示元素
        const warning = document.createElement('div');
        warning.className = 'slider-warning';
        warning.textContent = message;
        document.body.appendChild(warning);

        // 3秒后自动移除
        setTimeout(() => {
            if (warning.parentNode) {
                warning.parentNode.removeChild(warning);
            }
        }, 3000);
    }

    async adjustAngle(jointName, axis, angle, operation = 'add') {
        this.setLoading(true);

        try {
            const response = await fetch('/api/adjust', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    joint_name: jointName,
                    axis: axis,
                    angle: angle,
                    operation: operation
                })
            });

            const data = await response.json();
            if (data.success) {
                document.getElementById('renderImage').src = data.image;

                // 重新获取最新的角度值
                await this.loadCurrentAngles();

                this.updateFrameStatus(true);
            }
        } catch (error) {
            console.error('调节失败:', error);
        } finally {
            this.setLoading(false);
        }
    }

    updateSliderValue(jointName, axis, currentValue) {
        const valueInput = document.getElementById(`input_${jointName}_${axis}`);

        // 输入框显示绝对角度（2位小数）
        if (valueInput) {
            valueInput.value = currentValue.toFixed(2);
            valueInput.title = `当前绝对角度: ${currentValue.toFixed(2)}°`;
        }
    }

    async resetJoint(jointName, axis) {
        // 检查是否真的有修改
        const jointKey = `${jointName}_${axis}`;
        if (!this.modifiedJoints.has(jointKey)) {
            return; // 没有修改，不需要重置
        }

        this.setLoading(true);

        try {
            const response = await fetch('/api/reset_joint', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    joint_name: jointName,
                    axis: axis
                })
            });

            const data = await response.json();
            if (data.success) {
                document.getElementById('renderImage').src = data.image;

                // 重新获取最新的角度值
                await this.loadCurrentAngles();

                // 移除修改标记并更新按钮状态
                this.modifiedJoints.delete(jointKey);
                this.updateResetButtonState(jointName, axis, false);

                this.updateFrameStatus(data.modified);
            }
        } catch (error) {
            console.error('单项重置失败:', error);
        } finally {
            this.setLoading(false);
        }
    }

    async navigate(direction, targetIdx = null) {
        this.setLoading(true);

        try {
            const requestBody = {direction: direction};
            if (direction === 'goto' && targetIdx !== null) {
                requestBody.target_idx = targetIdx;
            }

            const response = await fetch('/api/navigate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();
            if (data.success) {
                // 更新前端的当前帧索引
                this.currentFrame = data.frame_idx;
                
                document.getElementById('renderImage').src = data.image;
                this.updateFrameInfo(data.frame_idx, data.frame_count);
                this.updateFrameStatus(data.modified);

                // 获取并显示当前帧的GT角度
                await this.loadCurrentAngles();
                
                // 导航时清除修改标记
                this.modifiedJoints.clear();
                this.updateAllResetButtonStates(false);
                
                // 确保数字按钮状态正确 - 立即更新高亮
                this.updateFrameButtonHighlight(data.frame_idx);
                
                // 确保数字按钮宽度正确
                setTimeout(() => {
                    this.adjustFrameNumbersWidth();
                }, 50);
            }
        } catch (error) {
            console.error('导航失败:', error);
        } finally {
            this.setLoading(false);
        }
    }



    /**
     * 更新所有重置按钮状态
     */
    updateAllResetButtonStates(isModified) {
        if (!this.jointConfigs) return;
        
        Object.keys(this.jointConfigs).forEach(jointName => {
            const config = this.jointConfigs[jointName];
            if (config && config.axes) {
                config.axes.forEach((_, axisIdx) => {
                    this.updateResetButtonState(jointName, axisIdx, isModified);
                });
            }
        });
    }



    async render() {
        this.setLoading(true);

        try {
            const response = await fetch('/api/render');
            const data = await response.json();

            if (data.success) {
                // 同步前端当前帧索引
                this.currentFrame = data.frame_idx;
                
                // 添加时间戳参数防止缓存
                const imageUrl = data.image + (data.timestamp ? `#${data.timestamp}` : `#${Date.now()}`);
                document.getElementById('renderImage').src = imageUrl;

                this.updateFrameInfo(data.frame_idx, data.frame_count);
                this.updateFrameStatus(data.modified);

                // 获取并显示当前帧的GT角度
                await this.loadCurrentAngles();

                // 图像加载完成后调整控制面板位置
                document.getElementById('renderImage').onload = () => {
                    setTimeout(() => {
                        this.forceAdjustPosition(); // 使用强制调整
                        // 只调整按钮宽度，不重新生成按钮（避免覆盖高亮状态）
                        this.adjustFrameNumbersWidth();
                        // 确保高亮状态正确
                        this.updateFrameButtonHighlight(this.currentFrame);
                    }, 100);
                };
            }
        } catch (error) {
            console.error('渲染失败:', error);
        } finally {
            this.setLoading(false);
        }
    }

    updateFrameInfo(frameIdx, frameCount) {
        // 只禁用复制上一帧按钮（当在第一帧时）
        const copyPrevBtn = document.getElementById('copyPrevBtn');
        if (copyPrevBtn) {
            copyPrevBtn.disabled = frameIdx === 0;
        }
        
        // 生成帧数字按钮
        this.generateFrameNumbers(frameIdx, frameCount);
    }

    updateFrameStatus(modified) {
        const indicator = document.getElementById('statusIndicator');
        if (indicator) {
            indicator.className = `status-indicator ${modified ? 'status-modified' : 'status-unmodified'}`;
        }
    }

    generateFrameNumbers(currentFrameIdx, frameCount) {
        const frameNumbersContainer = document.getElementById('frameNumbers');
        const renderImage = document.getElementById('renderImage');
        if (!frameNumbersContainer || !renderImage) return;

        // 清空现有按钮
        frameNumbersContainer.innerHTML = '';

        // 获取图片的实际显示宽度
        const imageDisplayWidth = renderImage.clientWidth;
        
        // 计算每个按钮的宽度（减去边框宽度）
        const buttonWidth = Math.floor(imageDisplayWidth / frameCount) - 1;
        
        // 设置容器宽度与图片宽度一致
        frameNumbersContainer.style.width = `${imageDisplayWidth}px`;

        // 为每一帧生成数字按钮
        for (let i = 0; i < frameCount; i++) {
            const btn = document.createElement('button');
            btn.className = 'frame-number-btn';
            btn.textContent = (i + 1).toString();
            btn.dataset.frameIndex = i.toString();
            btn.style.width = `${buttonWidth}px`;
            
            // 高亮当前帧
            if (i === currentFrameIdx) {
                btn.classList.add('active');
            }
            
            // 添加点击事件
            btn.addEventListener('click', () => this.jumpToFrame(i));
            
            frameNumbersContainer.appendChild(btn);
        }
    }

    async jumpToFrame(targetFrameIdx) {
        try {
            await this.navigate('goto', targetFrameIdx);
        } catch (error) {
            console.error('跳转到帧失败:', error);
        }
    }

    /**
     * 更新帧数字按钮的高亮状态（不重新生成按钮）
     */
    updateFrameButtonHighlight(currentFrameIdx) {
        const frameNumbersContainer = document.getElementById('frameNumbers');
        if (!frameNumbersContainer) return;

        const frameNumberBtns = frameNumbersContainer.querySelectorAll('.frame-number-btn');
        frameNumberBtns.forEach((btn, index) => {
            if (index === currentFrameIdx) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    adjustFrameNumbersWidth() {
        const frameNumbersContainer = document.getElementById('frameNumbers');
        const renderImage = document.getElementById('renderImage');
        if (!frameNumbersContainer || !renderImage) return;

        const frameNumberBtns = frameNumbersContainer.querySelectorAll('.frame-number-btn');
        if (frameNumberBtns.length === 0) return;

        // 获取图片的实际显示宽度
        const imageDisplayWidth = renderImage.clientWidth;
        
        // 如果图片宽度为0，等待图片加载
        if (imageDisplayWidth === 0) {
            setTimeout(() => this.adjustFrameNumbersWidth(), 100);
            return;
        }
        
        // 计算每个按钮的宽度
        const buttonWidth = Math.floor(imageDisplayWidth / frameNumberBtns.length) - 1;
        
        // 设置容器宽度与图片宽度一致
        frameNumbersContainer.style.width = `${imageDisplayWidth}px`;

        // 更新每个按钮的宽度
        frameNumberBtns.forEach(btn => {
            btn.style.width = `${buttonWidth}px`;
        });
    }

    checkAndAdjustFrameNumbers() {
        const frameNumbersContainer = document.getElementById('frameNumbers');
        const renderImage = document.getElementById('renderImage');
        
        if (!frameNumbersContainer || !renderImage) {
            // 如果元素还没有准备好，再次尝试
            setTimeout(() => this.checkAndAdjustFrameNumbers(), 200);
            return;
        }

        // 检查是否已经有数字按钮
        const frameNumberBtns = frameNumbersContainer.querySelectorAll('.frame-number-btn');
        if (frameNumberBtns.length > 0) {
            // 已经有按钮，调整宽度
            this.adjustFrameNumbersWidth();
        } else {
            // 还没有按钮，等待一会儿再检查
            setTimeout(() => this.checkAndAdjustFrameNumbers(), 200);
        }
    }

    resetSliders() {
        // 重置所有角度输入框到0.00
        document.querySelectorAll('.angle-input').forEach(input => {
            input.value = '0.00';
        });
    }

    setLoading(loading) {
        document.body.classList.toggle('loading', loading);
    }

    // 获取关节的调节范围限制
    getJointLimits(jointName, axisIdx) {
        const joints = {
            'global_orient': {'limits': [[-180, 180], [-90, 90], [-180, 180]]},
            'spine1': {'limits': [[-45, 45], [-30, 30], [-30, 30]]},
            'spine2': {'limits': [[-30, 30], [-25, 25], [-25, 25]]},
            'spine3': {'limits': [[-30, 30], [-25, 25], [-25, 25]]},
            'neck': {'limits': [[-60, 60], [-45, 45], [-45, 45]]},
            'left_hip': {'limits': [[-120, 120], [-45, 45], [-45, 45]]},
            'right_hip': {'limits': [[-120, 120], [-45, 45], [-45, 45]]},
            'left_knee': {'limits': [[0, 150], [-10, 10], [-10, 10]]},
            'right_knee': {'limits': [[0, 150], [-10, 10], [-10, 10]]},
            'left_ankle': {'limits': [[-30, 30], [-20, 20], [-15, 15]]},
            'right_ankle': {'limits': [[-30, 30], [-20, 20], [-15, 15]]},
            'left_shoulder': {'limits': [[-180, 180], [-90, 90], [-180, 180]]},
            'right_shoulder': {'limits': [[-180, 180], [-90, 90], [-180, 180]]},
            'left_elbow': {'limits': [[0, 150], [-90, 90], [-90, 90]]},
            'right_elbow': {'limits': [[0, 150], [-90, 90], [-90, 90]]},
            'left_wrist': {'limits': [[-60, 60], [-30, 30], [-45, 45]]},
            'right_wrist': {'limits': [[-60, 60], [-30, 30], [-45, 45]]}
        };
        return joints[jointName]?.limits?.[axisIdx];
    }

    // 显示输入框警告信息
    showInputWarning(inputElement, message) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'input-warning';
        warningDiv.textContent = message;
        inputElement.parentNode.insertBefore(warningDiv, inputElement.nextSibling);
        setTimeout(() => {
            warningDiv.remove();
        }, 3000); // 警告信息显示3秒
    }

    // 显示滑块警告信息
    showSliderWarning(message) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'slider-warning';
        warningDiv.textContent = message;
        document.getElementById('jointControls').appendChild(warningDiv);
        setTimeout(() => {
            warningDiv.remove();
        }, 3000); // 警告信息显示3秒
    }
}

// ==================== 序列参数更新功能 ====================

async function updateSequence() {
    const startFrame = parseInt(document.getElementById('startFrameInput').value);
    const frameInterval = parseInt(document.getElementById('frameIntervalInput').value);
    const numFrames = parseInt(document.getElementById('numFramesInput').value);
    const frameOffset = parseInt(document.getElementById('frameOffsetInput').value) || 0;

    // 显示加载状态
    const applyBtn = document.querySelector('.config-btn');
    const originalText = applyBtn.textContent;
    applyBtn.disabled = true;
    applyBtn.textContent = '应用中...';

    try {
        const response = await fetch('/api/update_sequence', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                start_frame: startFrame,
                frame_interval: frameInterval,
                num_frames: numFrames,
                frame_offset: frameOffset
            })
        });

        const data = await response.json();
        if (data.success) {
            // 显示成功消息
            const statusFloat = document.querySelector('.status-float');
            statusFloat.textContent = '✓ 参数更新成功';
            statusFloat.className = 'status-float show';
            
            // 强制清除图像缓存
            const renderImage = document.getElementById('renderImage');
            const currentSrc = renderImage.src;
            renderImage.src = ''; // 清除当前图像

            // 重新渲染并更新显示
            await app.render();

            // 再次强制刷新图像（防止缓存）
            setTimeout(() => {
                const newSrc = renderImage.src;
                if (newSrc === currentSrc) {
                    renderImage.src = newSrc + '&_t=' + Date.now();
                }
            }, 100);

            app.adjustControlPanelPosition();
            
            // 更新成功后重置原始参数值
            if (app) {
                app.resetOriginalParams();
            }
            
            // 隐藏状态提示
            setTimeout(() => {
                statusFloat.className = 'status-float';
            }, 3000);
        } else {
            // 显示错误消息
            const statusFloat = document.querySelector('.status-float');
            statusFloat.textContent = '✗ ' + data.error;
            statusFloat.className = 'status-float show error';
            
            setTimeout(() => {
                statusFloat.className = 'status-float';
            }, 5000);
            
            console.error('参数更新失败:', data.error);
        }
    } catch (error) {
        console.error('参数更新失败:', error);
        
        // 显示网络错误
        const statusFloat = document.querySelector('.status-float');
        statusFloat.textContent = '✗ 网络请求失败';
        statusFloat.className = 'status-float show error';
        
        setTimeout(() => {
            statusFloat.className = 'status-float';
        }, 5000);
    } finally {
        // 恢复按钮状态
        applyBtn.disabled = false;
        applyBtn.textContent = originalText;
    }
}

// ==================== 应用初始化 ====================

// 全局应用实例
let app = null;

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    app = new PoseAdjuster();
});

// 全局函数：重置参数配置
function resetParams() {
    if (!app) return;
    
    const paramInputs = [
        'startFrameInput',
        'frameIntervalInput', 
        'numFramesInput',
        'frameOffsetInput'
    ];

    // 将所有参数恢复到原始值
    paramInputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        if (input && app.originalParams[inputId]) {
            input.value = app.originalParams[inputId];
        }
    });

    // 更新按钮状态
    app.updateApplyButtonState();
}