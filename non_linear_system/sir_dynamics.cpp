#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
#define rep(i,a,b) for(int i=(a);i<(b);i++)

// 感染症のダイナミクスを理解し、予測するための数理モデルとしてのシミュレーションプログラム
// 時間と共にどのように感染者数が拡大し、収束していくかをシミュレート
// 大前提として、SIRモデルの数式としての微分方程式は把握すること

// ポイント：SIEモデルに基づく感染症のダイナミクスシミュレーションでは、感染者数Iは指数関数的に増加し、その後、回復者数Rの増加により減少するという挙動を示す
// ポイント２：感受性者数が最初最大、回復者が最小で、かつ、前者は指数関数的に減少すること、そして、後者は感染者数の増加に対して、増加していくこと、より感染拡大のピークとしてはかんん受精者の減少と感染者の増加が競合する事典が存在ー＞感染拡大のピーク
const double beta = 0.3; // 感染率
const double gamma =0.1; // 回復率
const double dt = 0.01; // タイムスタンプ
const int steps = 1000; // シミュレーションのステップ数
const double epsilon = 1e-6; // 収束判定の閾値

struct SIRState {
    double S; // 感受性者
    double I; // 感染者
    double R; // 回復者
};

SIRState computeNextState(const SIRState& currentState) {
    SIRState nextState;
    nextState.S = currentState.S - beta * currentState.S * currentState.I * dt;
    nextState.I = currentState.I + (beta * currentState.S * currentState.I - gamma *currentState.I) * dt;
    nextState.R = currentState.R + gamma * currentState.I * dt;
    return nextState;
}

int main() {
    // 初期状態
    SIRState state = { 0.99, 0.01, 0.0 }; // 初期感受性者、感染者、回復者

    vector<SIRState> states;
    states.push_back(state);

    int step = 0;
    while (state.I > epsilon) {
        state = computeNextState(state);
        states.push_back(state);
        step++;
    }

    // 下のループ分だと収束の判定およびそれまでのループが保証されないので上述の処理に変える
    // シミュレーション
    // rep(i,0,steps){
    //     state = computeNextState(state);
    //     states.push_back(state);
    // }

    // 結果を表示
    rep(i,0,states.size()) {
        cout<< "Time: "<< i * dt
            << " S: "<< states[i].S
            << " I: "<< states[i].I
            << " R: "<< states[i].R
            << endl;
    }

    cout << "Simulation ended after " << step << " steps. "<<endl;
    return 0;
}