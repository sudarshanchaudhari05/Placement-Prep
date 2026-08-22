/**
 * ==========================================================================
 * NEO-BRUTALIST SCIENTIFIC CALCULATOR - JAVASCRIPT ENGINE
 * ==========================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- STATE VARIABLES ---
  let expression = '';
  let lastResult = null;
  let isEvaluated = false;
  let angleMode = 'DEG'; // 'DEG' or 'RAD'
  let isSecondMode = false;
  let memoryValue = 0;
  let soundEnabled = true;
  let history = [];

  // --- DOM ELEMENTS ---
  const expressionDisplay = document.getElementById('expressionDisplay');
  const resultDisplay = document.getElementById('resultDisplay');
  const statusIndicator = document.getElementById('statusIndicator');
  const angleModeBtn = document.getElementById('angleModeBtn');
  const degLabel = document.getElementById('degLabel');
  const radLabel = document.getElementById('radLabel');
  const secondModeIndicator = document.getElementById('secondModeIndicator');
  const secondToggleBtn = document.getElementById('secondToggleBtn');
  const memoryIndicator = document.getElementById('memoryIndicator');
  const copyResultBtn = document.getElementById('copyResultBtn');
  const soundToggleBtn = document.getElementById('soundToggleBtn');
  const soundIcon = document.getElementById('soundIcon');
  const soundStatusText = document.getElementById('soundStatusText');
  const historyList = document.getElementById('historyList');
  const historyCountBadge = document.getElementById('historyCountBadge');
  const historyItemCount = document.getElementById('historyItemCount');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  const toastElement = document.getElementById('neoToast');
  const toastMessage = document.getElementById('toastMessage');
  const bsToast = new bootstrap.Toast(toastElement, { delay: 2200 });

  // --- WEB AUDIO API (MECHANICAL CLICK SYNTHESIS) ---
  let audioCtx = null;

  function playClickSound(type = 'default') {
    if (!soundEnabled) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      if (type === 'equals') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(520, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(780, audioCtx.currentTime + 0.08);
        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.1);
      } else if (type === 'clear') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(300, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(150, audioCtx.currentTime + 0.08);
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.09);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.09);
      } else {
        // Crisp mechanical click
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800 + Math.random() * 150, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(400, audioCtx.currentTime + 0.04);
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.045);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.05);
      }
    } catch (e) {
      console.warn('Audio Context error:', e);
    }
  }

  // --- TOAST HELPER ---
  function showToast(msg) {
    toastMessage.textContent = msg;
    bsToast.show();
  }

  // --- INITIALIZATION ---
  loadHistoryFromStorage();
  loadMemoryFromStorage();
  updateDisplay();

  // --- DISPLAY MANAGEMENT ---
  function updateDisplay(liveCalculate = true) {
    if (!expression) {
      expressionDisplay.textContent = '0';
      resultDisplay.textContent = lastResult !== null ? formatNumber(lastResult) : '0';
      statusIndicator.textContent = '● READY';
      statusIndicator.className = 'text-neo-lime fw-bold blink-dot';
      return;
    }

    expressionDisplay.textContent = formatDisplayExpression(expression);

    if (liveCalculate && !isEvaluated) {
      try {
        const preview = evaluateExpression(expression);
        if (!isNaN(preview) && isFinite(preview)) {
          resultDisplay.textContent = formatNumber(preview);
          statusIndicator.textContent = '● LIVE';
          statusIndicator.className = 'text-neo-cyan fw-bold';
        } else {
          statusIndicator.textContent = '● TYPING';
          statusIndicator.className = 'text-muted fw-bold';
        }
      } catch (e) {
        statusIndicator.textContent = '● TYPING';
        statusIndicator.className = 'text-muted fw-bold';
      }
    }

    // Auto scroll expression trail to the right
    expressionDisplay.parentElement.scrollLeft = expressionDisplay.parentElement.scrollWidth;
  }

  function formatDisplayExpression(expr) {
    return expr
      .replace(/\*/g, '×')
      .replace(/\//g, '÷')
      .replace(/-/g, '−')
      .replace(/sqrt\(/g, '√(')
      .replace(/cbrt\(/g, '∛(')
      .replace(/sin\(/g, 'sin(')
      .replace(/cos\(/g, 'cos(')
      .replace(/tan\(/g, 'tan(')
      .replace(/asin\(/g, 'sin⁻¹(')
      .replace(/acos\(/g, 'cos⁻¹(')
      .replace(/atan\(/g, 'tan⁻¹(')
      .replace(/log\(/g, 'log₁₀(')
      .replace(/ln\(/g, 'ln(')
      .replace(/exp\(/g, 'e^(')
      .replace(/tenpow\(/g, '10^(')
      .replace(/abs\(/g, 'abs(')
      .replace(/inv\(/g, '1/(');
  }

  function formatNumber(num) {
    if (typeof num !== 'number' || isNaN(num)) return 'Error';
    if (!isFinite(num)) return num > 0 ? 'Infinity' : '-Infinity';

    // Fix floating point issues e.g. 0.0000000000000001 -> 0
    const rounded = Number(Math.round(num + 'e12') + 'e-12');

    // Handle large/tiny numbers
    if (Math.abs(rounded) >= 1e12 || (Math.abs(rounded) > 0 && Math.abs(rounded) < 1e-6)) {
      return rounded.toExponential(6).replace('e+', 'e');
    }

    // Clean decimals
    const str = rounded.toString();
    if (str.includes('.')) {
      const parts = str.split('.');
      return parts[0] + '.' + parts[1].substring(0, 10);
    }
    return str;
  }

  // --- FACTORIAL HELPER ---
  function factorial(n) {
    if (n < 0) return NaN;
    if (n === 0 || n === 1) return 1;
    if (!Number.isInteger(n)) {
      // Lanczos Gamma function approximation for non-integers
      return gamma(n + 1);
    }
    if (n > 170) return Infinity; // JS upper limit
    let res = 1;
    for (let i = 2; i <= n; i++) res *= i;
    return res;
  }

  function gamma(z) {
    const g = 7;
    const C = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.1385710958311126, 9.9843695780195716e-6, 1.5056327351493116e-7
    ];
    if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
    z -= 1;
    let x = C[0];
    for (let i = 1; i < g + 2; i++) {
      x += C[i] / (z + i);
    }
    const t = z + g + 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
  }

  // --- SCIENTIFIC MATH PARSER & EVALUATOR ---
  function evaluateExpression(rawExpr) {
    if (!rawExpr || !rawExpr.trim()) return 0;

    let expr = rawExpr;

    // Replace unicode operators & constants
    expr = expr.replace(/×/g, '*');
    expr = expr.replace(/÷/g, '/');
    expr = expr.replace(/−/g, '-');
    expr = expr.replace(/π/g, 'Math.PI');
    expr = expr.replace(/(\b|\d)e(\b|[^\w])/g, '$1Math.E$2');

    // Handle percentage (e.g. 50% -> (50*0.01))
    expr = expr.replace(/(\d+(\.\d+)?)%/g, '($1*0.01)');

    // Handle Factorials (e.g. 5! or (3+2)!)
    while (expr.includes('!')) {
      const factMatch = expr.match(/(\((?:[^()]+)\)|\d+(?:\.\d+)?|\bMath\.PI\b|\bMath\.E\b)!/);
      if (!factMatch) break;
      expr = expr.replace(factMatch[0], `factorial(${factMatch[1]})`);
    }

    // Handle Power operator ^ -> **
    expr = expr.replace(/\^/g, '**');

    // Handle Implicit Multiplication:
    // 1. Number before parenthesis: 3(4) -> 3*(4)
    expr = expr.replace(/(\d)(\()/g, '$1*$2');
    // 2. Parenthesis before number: (4)3 -> (4)*3
    expr = expr.replace(/(\))(\d)/g, '$1*$2');
    // 3. Parenthesis before parenthesis: (2)(3) -> (2)*(3)
    expr = expr.replace(/(\))(\()/g, '$1*$2');
    // 4. Number before function/constant: 2sin(30) -> 2*sin(30), 2Math.PI -> 2*Math.PI
    expr = expr.replace(/(\d)(sin|cos|tan|asin|acos|atan|log|ln|sqrt|cbrt|abs|inv|exp|tenpow|Math\.PI|Math\.E)/g, '$1*$2');
    // 5. Constant before parenthesis: Math.PI(2) -> Math.PI*(2)
    expr = expr.replace(/(Math\.PI|Math\.E)(\()/g, '$1*$2');
    // 6. Parenthesis before function/constant: (2)Math.PI -> (2)*Math.PI
    expr = expr.replace(/(\))(Math\.PI|Math\.E|sin|cos|tan|asin|acos|atan|log|ln|sqrt|cbrt)/g, '$1*$2');

    // Scope Functions
    const isDeg = angleMode === 'DEG';
    const toRad = deg => (deg * Math.PI) / 180;
    const toDeg = rad => (rad * 180) / Math.PI;

    const mathScope = {
      Math: Math,
      factorial: factorial,
      sin: x => isDeg ? Math.sin(toRad(x)) : Math.sin(x),
      cos: x => isDeg ? Math.cos(toRad(x)) : Math.cos(x),
      tan: x => {
        const rad = isDeg ? toRad(x) : x;
        if (Math.abs(Math.cos(rad)) < 1e-15) throw new Error('Undefined (tan 90°)');
        return Math.tan(rad);
      },
      asin: x => {
        if (x < -1 || x > 1) throw new Error('Domain Error');
        const val = Math.asin(x);
        return isDeg ? toDeg(val) : val;
      },
      acos: x => {
        if (x < -1 || x > 1) throw new Error('Domain Error');
        const val = Math.acos(x);
        return isDeg ? toDeg(val) : val;
      },
      atan: x => {
        const val = Math.atan(x);
        return isDeg ? toDeg(val) : val;
      },
      log: x => {
        if (x <= 0) throw new Error('Domain Error');
        return Math.log10(x);
      },
      ln: x => {
        if (x <= 0) throw new Error('Domain Error');
        return Math.log(x);
      },
      sqrt: x => {
        if (x < 0) throw new Error('Domain Error');
        return Math.sqrt(x);
      },
      cbrt: x => Math.cbrt(x),
      abs: x => Math.abs(x),
      exp: x => Math.exp(x),
      tenpow: x => Math.pow(10, x),
      sqr: x => Math.pow(x, 2),
      cube: x => Math.pow(x, 3),
      inv: x => {
        if (x === 0) throw new Error('Division by Zero');
        return 1 / x;
      }
    };

    // Safe execution using scoped Function
    const funcArgs = Object.keys(mathScope);
    const funcVals = Object.values(mathScope);
    const evaluator = new Function(...funcArgs, `return (${expr});`);
    const res = evaluator(...funcVals);

    // Clean up precision artifacts like sin(180) = 1.22e-16 -> 0
    if (typeof res === 'number' && Math.abs(res) < 1e-14 && Math.abs(res) > 0) {
      return 0;
    }

    return res;
  }

  // --- ACTIONS & OPERATOR HANDLERS ---
  function appendValue(val) {
    playClickSound();

    if (isEvaluated) {
      // If user types an operator after calculation, continue with last result
      if (['+', '-', '*', '/', '%', '^'].includes(val)) {
        expression = (lastResult !== null ? lastResult.toString() : '0') + val;
      } else {
        expression = val;
      }
      isEvaluated = false;
    } else {
      if (expression === '0' && !['.', '+', '-', '*', '/', '%', '^'].includes(val)) {
        expression = val;
      } else {
        expression += val;
      }
    }
    updateDisplay();
  }

  function appendFunction(fnName) {
    playClickSound();

    if (isEvaluated) {
      expression = `${fnName}(${lastResult !== null ? lastResult : ''}`;
      isEvaluated = false;
    } else {
      if (expression === '0') {
        expression = `${fnName}(`;
      } else {
        expression += `${fnName}(`;
      }
    }
    updateDisplay();
  }

  function clearAll() {
    playClickSound('clear');
    expression = '';
    lastResult = null;
    isEvaluated = false;
    updateDisplay();
    showToast('Cleared All');
  }

  function backspace() {
    playClickSound();
    if (isEvaluated) {
      expression = '';
      isEvaluated = false;
    } else if (expression.length > 0) {
      // Check if ending with a function name like 'sin(' or 'log10('
      const fnMatches = ['asin(', 'acos(', 'atan(', 'sin(', 'cos(', 'tan(', 'log(', 'ln(', 'sqrt(', 'cbrt(', 'abs(', 'inv(', 'exp(', 'tenpow('];
      let removed = false;
      for (const fn of fnMatches) {
        if (expression.endsWith(fn)) {
          expression = expression.slice(0, -fn.length);
          removed = true;
          break;
        }
      }
      if (!removed) {
        expression = expression.slice(0, -1);
      }
    }
    updateDisplay();
  }

  function toggleSign() {
    playClickSound();
    if (isEvaluated && lastResult !== null) {
      lastResult = -lastResult;
      expression = lastResult.toString();
      isEvaluated = false;
      updateDisplay();
      return;
    }

    if (!expression) {
      expression = '-';
      updateDisplay();
      return;
    }

    // Toggle negation on current expression
    if (expression.startsWith('-(') && expression.endsWith(')')) {
      expression = expression.substring(2, expression.length - 1);
    } else {
      expression = `-(${expression})`;
    }
    updateDisplay();
  }

  function evaluate() {
    if (!expression && lastResult === null) return;
    playClickSound('equals');

    try {
      // Auto-close missing unclosed parentheses
      let openCount = (expression.match(/\(/g) || []).length;
      let closeCount = (expression.match(/\)/g) || []).length;
      while (openCount > closeCount) {
        expression += ')';
        closeCount++;
      }

      const calculated = evaluateExpression(expression);

      if (isNaN(calculated)) {
        throw new Error('Invalid calculation');
      }

      const formattedResult = formatNumber(calculated);

      // Save to history
      saveToHistory(expression, formattedResult);

      lastResult = calculated;
      resultDisplay.textContent = formattedResult;
      isEvaluated = true;
      statusIndicator.textContent = '● SUCCESS';
      statusIndicator.className = 'text-neo-lime fw-bold';

    } catch (err) {
      resultDisplay.textContent = err.message || 'Syntax Error';
      statusIndicator.textContent = '● ERROR';
      statusIndicator.className = 'text-neo-pink fw-bold';
      isEvaluated = true;
    }
  }

  // --- SECOND MODE TOGGLE ---
  function toggleSecondMode() {
    playClickSound();
    isSecondMode = !isSecondMode;

    const primaryLabels = document.querySelectorAll('.fn-primary');
    const invLabels = document.querySelectorAll('.fn-inv');

    if (isSecondMode) {
      secondToggleBtn.classList.add('bg-neo-lime', 'text-dark');
      secondToggleBtn.classList.remove('bg-neo-purple', 'text-white');
      secondModeIndicator.textContent = '2nd: ON';
      secondModeIndicator.classList.add('pill-active');

      primaryLabels.forEach(el => el.classList.add('d-none'));
      invLabels.forEach(el => el.classList.remove('d-none'));
    } else {
      secondToggleBtn.classList.remove('bg-neo-lime', 'text-dark');
      secondToggleBtn.classList.add('bg-neo-purple', 'text-white');
      secondModeIndicator.textContent = '2nd: OFF';
      secondModeIndicator.classList.remove('pill-active');

      primaryLabels.forEach(el => el.classList.remove('d-none'));
      invLabels.forEach(el => el.classList.add('d-none'));
    }
  }

  // --- ANGLE MODE TOGGLE ---
  function toggleAngleMode() {
    playClickSound();
    if (angleMode === 'DEG') {
      angleMode = 'RAD';
      degLabel.classList.remove('fw-bold');
      degLabel.classList.add('opacity-50');
      radLabel.classList.add('fw-bold');
      radLabel.classList.remove('opacity-50');
      showToast('Angle Mode: RADIANS');
    } else {
      angleMode = 'DEG';
      degLabel.classList.add('fw-bold');
      degLabel.classList.remove('opacity-50');
      radLabel.classList.remove('fw-bold');
      radLabel.classList.add('opacity-50');
      showToast('Angle Mode: DEGREES');
    }
    updateDisplay();
  }

  // --- MEMORY OPERATIONS ---
  function handleMemory(action) {
    playClickSound();
    const currentVal = (lastResult !== null && isEvaluated) ? lastResult : (evaluateExpression(expression) || 0);

    switch (action) {
      case 'memory-clear':
        memoryValue = 0;
        localStorage.setItem('neocalc_memory', 0);
        updateMemoryDisplay();
        showToast('Memory Cleared (MC)');
        break;

      case 'memory-recall':
        appendValue(memoryValue.toString());
        showToast(`Recalled Memory: ${memoryValue}`);
        break;

      case 'memory-add':
        memoryValue += currentVal;
        localStorage.setItem('neocalc_memory', memoryValue);
        updateMemoryDisplay();
        showToast(`Added to Memory: M+`);
        break;

      case 'memory-sub':
        memoryValue -= currentVal;
        localStorage.setItem('neocalc_memory', memoryValue);
        updateMemoryDisplay();
        showToast(`Subtracted from Memory: M-`);
        break;
    }
  }

  function updateMemoryDisplay() {
    memoryIndicator.textContent = `M: ${formatNumber(memoryValue)}`;
    if (memoryValue !== 0) {
      memoryIndicator.classList.add('pill-active');
    } else {
      memoryIndicator.classList.remove('pill-active');
    }
  }

  function loadMemoryFromStorage() {
    const saved = localStorage.getItem('neocalc_memory');
    if (saved) {
      memoryValue = parseFloat(saved) || 0;
    }
    updateMemoryDisplay();
  }

  // --- CALCULATION HISTORY ---
  function saveToHistory(expr, result) {
    const item = {
      id: Date.now(),
      expression: expr,
      result: result,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };

    history.unshift(item);
    if (history.length > 50) history.pop(); // Limit history to 50 items

    localStorage.setItem('neocalc_history', JSON.stringify(history));
    renderHistory();
  }

  function renderHistory() {
    historyCountBadge.textContent = history.length;
    historyItemCount.textContent = history.length;

    if (history.length === 0) {
      historyList.innerHTML = `
        <div class="neo-box bg-white p-3 text-center text-muted font-jetbrains fs-7 empty-history-msg">
          <i class="bi bi-calculator display-6 d-block mb-2 text-dark"></i>
          No calculations yet.<br>Do some math to see history here!
        </div>
      `;
      return;
    }

    historyList.innerHTML = history.map(item => `
      <div class="history-item neo-box p-2" data-expr="${encodeURIComponent(item.expression)}" data-res="${item.result}">
        <div class="d-flex justify-content-between text-muted fs-8 font-jetbrains mb-1">
          <span>${item.timestamp}</span>
          <span class="badge bg-neo-yellow text-dark border border-dark">Click to load</span>
        </div>
        <div class="font-jetbrains text-truncate text-secondary fs-7">${formatDisplayExpression(item.expression)}</div>
        <div class="font-jetbrains fw-bold fs-5 text-dark">= ${item.result}</div>
      </div>
    `).join('');

    // Attach click handlers to reload history into calculator
    historyList.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => {
        playClickSound();
        const rawExpr = decodeURIComponent(el.getAttribute('data-expr'));
        const res = el.getAttribute('data-res');
        expression = rawExpr;
        lastResult = parseFloat(res);
        isEvaluated = true;
        updateDisplay();
        showToast('Loaded calculation from history');
        
        // Auto-close offcanvas on selection
        const offcanvasEl = document.getElementById('historyOffcanvas');
        const offcanvas = bootstrap.Offcanvas.getInstance(offcanvasEl);
        if (offcanvas) offcanvas.hide();
      });
    });
  }

  function loadHistoryFromStorage() {
    try {
      const saved = localStorage.getItem('neocalc_history');
      if (saved) {
        history = JSON.parse(saved);
      }
    } catch (e) {
      history = [];
    }
    renderHistory();
  }

  function clearHistory() {
    playClickSound('clear');
    history = [];
    localStorage.removeItem('neocalc_history');
    renderHistory();
    showToast('History Cleared');
  }

  // --- SOUND TOGGLE ---
  function toggleSound() {
    soundEnabled = !soundEnabled;
    if (soundEnabled) {
      soundIcon.className = 'bi bi-volume-up-fill';
      soundStatusText.textContent = 'SFX: ON';
      soundToggleBtn.classList.remove('bg-neo-gray');
      soundToggleBtn.classList.add('bg-neo-cyan');
      showToast('Sound Effects ON');
      playClickSound();
    } else {
      soundIcon.className = 'bi bi-volume-mute-fill';
      soundStatusText.textContent = 'SFX: OFF';
      soundToggleBtn.classList.remove('bg-neo-cyan');
      soundToggleBtn.classList.add('bg-neo-gray');
      showToast('Sound Effects OFF');
    }
  }

  // --- COPY RESULT ---
  function copyResult() {
    const text = resultDisplay.textContent;
    if (text && text !== 'Error' && text !== 'Syntax Error') {
      navigator.clipboard.writeText(text).then(() => {
        showToast(`Copied ${text} to clipboard!`);
      }).catch(() => {
        showToast('Failed to copy');
      });
    }
  }

  // --- BUTTON EVENT LISTENERS ---
  document.querySelectorAll('button[data-val]').forEach(btn => {
    btn.addEventListener('click', () => {
      const val = btn.getAttribute('data-val');
      appendValue(val);
    });
  });

  document.querySelectorAll('button[data-fn]').forEach(btn => {
    btn.addEventListener('click', () => {
      const primaryFn = btn.getAttribute('data-fn');
      const invFn = btn.getAttribute('data-inv');
      const chosenFn = (isSecondMode && invFn) ? invFn : primaryFn;

      if (chosenFn === 'fact') {
        appendValue('!');
      } else if (chosenFn === 'pow') {
        appendValue('^');
      } else if (chosenFn === 'yroot') {
        appendValue('^(1/');
      } else if (chosenFn === 'sqr') {
        appendValue('^2');
      } else if (chosenFn === 'cube') {
        appendValue('^3');
      } else {
        appendFunction(chosenFn);
      }
    });
  });

  document.querySelectorAll('button[data-action]').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.getAttribute('data-action');
      if (action === 'clear-all') clearAll();
      else if (action === 'backspace') backspace();
      else if (action === 'toggle-sign') toggleSign();
      else if (action === 'evaluate') evaluate();
      else if (action.startsWith('memory-')) handleMemory(action);
    });
  });

  // Mode & Audio Buttons
  angleModeBtn.addEventListener('click', toggleAngleMode);
  secondToggleBtn.addEventListener('click', toggleSecondMode);
  soundToggleBtn.addEventListener('click', toggleSound);
  clearHistoryBtn.addEventListener('click', clearHistory);
  copyResultBtn.addEventListener('click', copyResult);

  // --- KEYBOARD LISTENER ---
  window.addEventListener('keydown', (e) => {
    // Ignore keys if focus is in an input or modal dialog
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') return;

    const key = e.key;

    // Highlight corresponding visual button
    let matchingBtn = document.querySelector(`button[data-key="${key}"]`);
    if (matchingBtn) {
      matchingBtn.classList.add('btn-active-flash');
      setTimeout(() => matchingBtn.classList.remove('btn-active-flash'), 120);
    }

    if (key >= '0' && key <= '9') {
      appendValue(key);
    } else if (['+', '-', '*', '/', '.', '(', ')', '%', '^', '!'].includes(key)) {
      appendValue(key);
    } else if (key === 'Enter' || key === '=') {
      e.preventDefault();
      evaluate();
    } else if (key === 'Backspace') {
      e.preventDefault();
      backspace();
    } else if (key === 'Escape') {
      e.preventDefault();
      clearAll();
    } else if (key.toLowerCase() === 's') {
      appendFunction(isSecondMode ? 'asin' : 'sin');
    } else if (key.toLowerCase() === 'c') {
      appendFunction(isSecondMode ? 'acos' : 'cos');
    } else if (key.toLowerCase() === 't') {
      appendFunction(isSecondMode ? 'atan' : 'tan');
    } else if (key.toLowerCase() === 'l') {
      appendFunction(isSecondMode ? 'tenpow' : 'log');
    } else if (key.toLowerCase() === 'n') {
      appendFunction(isSecondMode ? 'exp' : 'ln');
    } else if (key.toLowerCase() === 'p') {
      appendValue('π');
    } else if (key.toLowerCase() === 'e') {
      appendValue('e');
    } else if (key.toLowerCase() === 'd') {
      toggleAngleMode();
    }
  });
});
