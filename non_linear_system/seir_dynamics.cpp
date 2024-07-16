// SEIRモデルはSIRモデルを拡張し、潜伏期間を考慮するために「潜伏期感染者(Exposed)」という状態を追加したプログラム
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
#define rep(i,a,b) for(int i=(a);i<(b);i++)

// SEIRモデルに基づく感染症ダイナミクスのシミュレーションプログラム

const double beta = 0.3;
const double gamma =0.1;
const double sigma = 0.2; // 潜伏期間の逆数
const double dt = 0.01;
const double epsilon = 1e-6;

// SEIRモデルの状態をタイムスタンプごとに保持するための構造体
struct SEIRState {
    double S; // 感受性者数（Susceptible）
    double E; // 潜伏期感染者数（Exposed）
    double I; // 感染者数（Infectious）
    double R; // 回復者数（Recovered）
};

// 次の状態を計算する関数
// 各保持する変数の値について差分に寄与する要素の減少要因と増加要因を訂正的に一般の数式化された部分から導出して、時間の差分も考慮に入れるようにモデリング
SEIRState computeNextState(const SEIRState& currentState) {
    SEIRState nextState;
    nextState.S = currentState.S - beta * currentState.S * currentState.I * dt;
    nextState.E = currentState.E + (beta * currentState.S * currentState.I - sigma * currentState.E) * dt;
    nextState.I = currentState.I + (sigma * currentState.E - gamma * currentState.I) *dt;
    nextState.R = currentState.R + gamma * currentState.I *dt;
    return nextState;
}

int main () {
    // 初期状態として各対象の母集団に対する割合を決める
    SEIRState state = { 0.99, 0.01, 0.0, 0.0 }; // 初期感受性者、潜伏期間者、感染者、回復者

    vector<SEIRState> states;
    states.push_back(state);

    // シミュレーション
    int step = 0;
    while (state.I > epsilon || state.E > epsilon)
    {
        state = computeNextState(state);
        states.push_back(state);
        step++;
    }

// 結果を表示
    rep(i, 0, states.size()) {
        cout << "Time: " << i * dt
            << " S: " << states[i].S
            << " E: " << states[i].E
            << " I: " << states[i].I
            << " R: " << states[i].R
            << endl;
    }
    cout << "Simulation ended after " << step << " steps." << endl;
    return 0;
}
