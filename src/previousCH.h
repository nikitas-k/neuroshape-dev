#include <stdbool.h>
#include <math.h>

#define PI 3.14159265
#define ToleranceOfConvexAngle 0.001
#define LENGTH_EPSILON_CONTROL 0.001

typedef struct Window {
    bool fIsOnLeftSubtree;
    bool fParentIsPseudoSource;
    bool fDirectParentEdgeOnLeft;
    bool fDirectParenIsPseudoSource;
    char birthTimeOfParent;
    int indexOfParent;
    int indexOfRoot;
    int indexOfCurEdge;
    int level;
    double disToRoot;
    double proportions[2];
    double entryPropOfParent;
    double leftLen;
    double rightLen;
    double xUponUnfolding;
    double yUponUnfolding;
} Window;

bool IsTooNarrowWindow(const Window* w) {
    return w->proportions[1] - w->proportions[0] < LENGTH_EPSILON_CONTROL;
}

void ComputeTheOnlyLeftChild(const Window* w, double disToAngle);
void ComputeTheOnlyRightChild(const Window* w, double disToAngle);
void ComputeTheOnlyLeftTrimmedChild(const Window* w, double disToAngle);
void ComputeTheOnlyRightTrimmedChild(const Window* w, double disToAngle);
void ComputeLeftTrimmedChildWithParent(const Window* w, double disToAngle);
void ComputeRightTrimmedChildWithParent(const Window* w, double disToAngle);

void ComputeChildrenOfWindow(const Window* w, double disToAngle) {
    if (w->fIsOnLeftSubtree) {
        ComputeTheOnlyLeftChild(w, disToAngle);
    } else {
        ComputeTheOnlyRightChild(w, disToAngle);
    }
}

void ComputeTheOnlyLeftChild(const Window* w, double disToAngle) {
    if (w->fDirectParentEdgeOnLeft) {
        ComputeLeftTrimmedChildWithParent(w, disToAngle);
    } else {
        ComputeTheOnlyLeftTrimmedChild(w, disToAngle);
    }
}

void ComputeTheOnlyRightChild(const Window* w, double disToAngle) {
    if (w->fDirectParentEdgeOnLeft) {
        ComputeTheOnlyRightTrimmedChild(w, disToAngle);
    } else {
        ComputeRightTrimmedChildWithParent(w, disToAngle);
    }
}

void ComputeTheOnlyLeftTrimmedChild(const Window* w, double disToAngle) {
    Window quoteW;
    quoteW.fParentIsPseudoSource = w->fParentIsPseudoSource;
    quoteW.fDirectParenIsPseudoSource = false;
    quoteW.fDirectParentEdgeOnLeft = true;
    quoteW.indexOfCurEdge = w->indexOfCurEdge;
    quoteW.disToRoot = w->disToRoot;
    quoteW.leftLen = w->leftLen;
    quoteW.rightLen = disToAngle;
    if (!CheckValidityOfWindow(&quoteW)) {
        return;
    }
    quoteW.level = w->level + 1;
    quoteW.birthTimeOfParent = w->birthTimeOfParent;
    quoteW.indexOfParent = w->indexOfParent;
    quoteW.indexOfRoot = w->indexOfRoot;
    quoteW.fIsOnLeftSubtree = w->fIsOnLeftSubtree;
    quoteW.entryPropOfParent = w->entryPropOfParent;
    ComputeChildrenOfWindow(&quoteW, disToAngle);
}

void ComputeTheOnlyRightTrimmedChild(const Window* w, double disToAngle) {
    Window quoteW;
    quoteW.fParentIsPseudoSource = w->fParentIsPseudoSource;
    quoteW.fDirectParenIsPseudoSource = false;
    quoteW.fDirectParentEdgeOnLeft = false;
    quoteW.indexOfCurEdge = w->indexOfCurEdge;
    quoteW.disToRoot = w->disToRoot;
    quoteW.leftLen = disToAngle;
    quoteW.rightLen = w->rightLen;
    if (!CheckValidityOfWindow(&quoteW)) {
        return;
    }
    quoteW.level = w->level + 1;
    quoteW.birthTimeOfParent = w->birthTimeOfParent;
    quoteW.indexOfParent = w->indexOfParent;
    quoteW.indexOfRoot = w->indexOfRoot;
    quoteW.fIsOnLeftSubtree = w->fIsOnLeftSubtree;
    quoteW.entryPropOfParent = w->entryPropOfParent;
    ComputeChildrenOfWindow(&quoteW, disToAngle);
}

void ComputeLeftTrimmedChildWithParent(const Window* w, double disToAngle) {
    Window quoteW;
    quoteW.fParentIsPseudoSource = false;
    quoteW.fDirectParenIsPseudoSource = false;
    quoteW.fDirectParentEdgeOnLeft = true;
    quoteW.indexOfCurEdge = w->indexOfCurEdge;
    quoteW.disToRoot = w->disToRoot;
    quoteW.leftLen = w->leftLen;
    quoteW.rightLen = disToAngle;
    if (!CheckValidityOfWindow(&quoteW)) {
        return;
    }
    quoteW.level = w->level + 1;
    quoteW.birthTimeOfParent = w->birthTimeOfParent;
    quoteW.indexOfParent = w->indexOfCurEdge;
    quoteW.indexOfRoot = w->indexOfRoot;
    quoteW.fIsOnLeftSubtree = true;
    quoteW.entryPropOfParent = w->entryPropOfParent;
    ComputeChildrenOfWindow(&quoteW, disToAngle);
}

void ComputeRightTrimmedChildWithParent(const Window* w, double disToAngle) {
    Window quoteW;
    quoteW.fParentIsPseudoSource = false;
    quoteW.fDirectParenIsPseudoSource = false;
    quoteW.fDirectParentEdgeOnLeft = false;
    quoteW.indexOfCurEdge = w->indexOfCurEdge;
    quoteW.disToRoot = w->disToRoot;
    quoteW.leftLen = disToAngle;
    quoteW.rightLen = w->rightLen;
    if (!CheckValidityOfWindow(&quoteW)) {
        return;
    }
    quoteW.fIsOnLeftSubtree = false;
    quoteW.birthTimeOfParent = w->birthTimeOfParent;
    quoteW.indexOfParent = w->indexOfCurEdge;
    quoteW.indexOfRoot = w->indexOfRoot;
    quoteW.level = w->level + 1;
    quoteW.entryPropOfParent = w->entryPropOfParent;
    ComputeChildrenOfWindow(&quoteW, disToAngle);
}
