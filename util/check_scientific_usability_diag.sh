#!/bin/bash

# =============================================================
# OpenFOAM 模拟可用性评估 4.3 (增加残差绝对量级拦截，防高位震荡)
# =============================================================

LOG_FILE_PATH=${1:-"log.simpleFoam"}

# ----------------- 评估标准配置区 -----------------
# 压力残差的绝对容忍上限 (根据具体算例可调，建议 0.01 ~ 0.001)
MAX_P_RESIDUAL=0.01
# --------------------------------------------------

# 自动获取 Case 所在的根目录
CASE_DIR=$(cd "$(dirname "$LOG_FILE_PATH")"; pwd)
LOG_FILE_NAME=$(basename "$LOG_FILE_PATH")
FLUX_DIR="$CASE_DIR/postProcessing"
LOGS_DIR="$CASE_DIR/logs"

echo "========================================================="
echo "📊 评估目录: $CASE_DIR"
echo "📊 日志文件: $LOG_FILE_NAME"
echo "========================================================="

# 1. 运行状态检查
if [ ! -f "$CASE_DIR/$LOG_FILE_NAME" ]; then
    echo "❌ 错误: 找不到日志文件 $LOG_FILE_PATH"
    exit 1
fi

# 2. 物理守恒检查
echo "📈 [1/2] 质量守恒验证 (Mass Flux Balance):"
TOTAL_IN=0
TOTAL_OUT=0
FOUND_FLUX=false
USABLE_FLUX=false

if [ -d "$FLUX_DIR" ]; then
    FLUX_FILES=$(find "$FLUX_DIR" -name "surfaceFieldValue.dat" | grep "phi")
    
    if [ -z "$FLUX_FILES" ]; then
        echo "   ❓ 警告: 未找到 phi 相关数据。"
    else
        for f in $FLUX_FILES; do
            NAME=$(echo "$f" | grep -o "phi_[^/]*")
            VAL=$(tail -n 1 "$f" | awk '{print $NF}' | tr -d '\r')
            
            if [[ ! "$VAL" =~ ^[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]]; then
                continue
            fi

            IS_POSITIVE=$(awk -v v="$VAL" 'BEGIN {print (v > 0 ? 1 : 0)}')
            if [ "$IS_POSITIVE" -eq 1 ]; then
                TOTAL_IN=$(awk -v t="$TOTAL_IN" -v v="$VAL" 'BEGIN {printf "%.10e", t+v}')
            else
                TOTAL_OUT=$(awk -v t="$TOTAL_OUT" -v v="$VAL" 'BEGIN {printf "%.10e", t+v}')
            fi
            FOUND_FLUX=true
            printf "   - %-10s: %e\n" "$NAME" "$VAL"
        done
    fi
fi

if [ "$FOUND_FLUX" = true ]; then
    NET_FLUX=$(awk -v tin="$TOTAL_IN" -v tout="$TOTAL_OUT" 'BEGIN {printf "%.10e", tin+tout}')
    
    ERR=$(awk -v net="$NET_FLUX" -v tin="$TOTAL_IN" 'BEGIN {
        abs_in = (tin < 0) ? -tin : tin;
        if (abs_in == 0) {
            print "0.000000";
        } else {
            err = (net / abs_in) * 100;
            if (err < 0) err = -err;
            printf "%.6f", err;
        }
    }')
    
    printf "   >> 净通量误差: %s%%\n" "$ERR"
    
    IS_CONVERGED=$(awk -v err="$ERR" 'BEGIN {print (err < 1.0 ? 1 : 0)}')
    if [ "$IS_CONVERGED" -eq 1 ]; then
        echo "   ✅ 守恒评价: 质量守恒通过 (误差 < 1.0%)。"
        USABLE_FLUX=true
    else
        echo "   ❌ 评价: 质量不守恒。"
        USABLE_FLUX=false
    fi
else
    echo "   ❌ 失败: 无法定位通量数据。"
fi

# 3. 残差趋势检查
echo ""
echo "📉 [2/2] 流场定型检查 (Trend Stability & Magnitude):"
USABLE_TREND=false
P_TREND=""

if [ -d "$LOGS_DIR" ]; then
    P_FILE=$(ls "$LOGS_DIR"/p_[0-9]* "$LOGS_DIR"/pcorr_[0-9]* 2>/dev/null | head -n 1)
    
    if [ -f "$P_FILE" ]; then
        echo "   📂 发现 logs 目录，已从 $(basename "$P_FILE") 极速提取残差。"
        P_TREND=$(tail -n 200 "$P_FILE" | awk '{print $2}' | tr -d '\r' | xargs)
    fi
fi

if [ -z "$P_TREND" ]; then
    echo "   🔍 未检测到独立的残差日志，正在解析主日志文件 (耗时可能较长)..."
    P_TREND=$(grep -E "Solving for p(_[0-9]+)?|Solving for pcorr" "$CASE_DIR/$LOG_FILE_NAME" | tail -n 200 | awk -F'initial residual = ' '{print $2}' | awk -F',' '{print $1}' | tr -d '\r' | xargs)
fi

if [ -n "$P_TREND" ]; then
    P_ARRAY=($P_TREND)
    COUNT=${#P_ARRAY[@]}
    
    if [ "$COUNT" -gt 0 ]; then
        LAST_P=${P_ARRAY[$((COUNT-1))]}
        
        SUM=0
        for i in "${P_ARRAY[@]}"; do 
            SUM=$(awk -v s="$SUM" -v i="$i" 'BEGIN {printf "%.10e", s+i}')
        done
        AVG_P=$(awk -v s="$SUM" -v c="$COUNT" 'BEGIN {printf "%.10e", s/c}')
        
        DEV=$(awk -v lp="$LAST_P" -v ap="$AVG_P" 'BEGIN {
            if (ap == 0) {
                print "0.0000";
            } else {
                dev = (lp - ap) / ap;
                if (dev < 0) dev = -dev;
                printf "%.4f", dev;
            }
        }')
        
        printf "   >> 最后压力残差: %e (门槛: %s)\n" "$LAST_P" "$MAX_P_RESIDUAL"
        
        # 核心修复点：双重检查 (绝对量级 + 相对波动)
        IS_STABLE=$(awk -v dev="$DEV" 'BEGIN {print (dev < 0.2 ? 1 : 0)}')
        IS_CONVERGED_ABS=$(awk -v lp="$LAST_P" -v maxp="$MAX_P_RESIDUAL" 'BEGIN {print (lp < maxp ? 1 : 0)}')

        if [ "$IS_CONVERGED_ABS" -eq 0 ]; then
            echo "   ❌ 评价: 压力残差过大 ($LAST_P > $MAX_P_RESIDUAL)，流场处于高位震荡或未收敛状态！"
            USABLE_TREND=false
        elif [ "$IS_STABLE" -eq 1 ]; then
            echo "   ✅ 趋势评价: 残差已平稳 (波动 < 20%) 且满足量级要求。"
            USABLE_TREND=true
        else
            echo "   ⚠️ 评价: 压力残差量级达标，但仍在剧烈波动 (波动率: $DEV)。"
            USABLE_TREND=false
        fi
    fi
else
    echo "   ❓ 无法解析残差。日志中可能不包含 initial residual 信息。"
fi

# 4. 结论总结
echo ""
if [ "$USABLE_FLUX" = true ] && [ "$USABLE_TREND" = true ]; then
    echo "🌟 综合判定: 该实验【可用】！"
else
    echo "🚩 综合判定: 该实验【存疑或不可用】。"
fi
echo "========================================================="