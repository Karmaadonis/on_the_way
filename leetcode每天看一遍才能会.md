# 不做笔记是不行的，做了也不一定行，但是下面的是一直不行，所以建议每天看一遍


# 散题
## 重复的子字符串

$给定一个非空的字符串 s ，检查是否可以通过由它的一个子串重复多次构成。$

``` c++
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        for (int i = 1; i * 2 <= n; ++i) {
            if (n % i == 0) {
                bool match = true;
                for (int j = i; j < n; ++j) {
                    if (s[j] != s[j - i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    return true;
                }
            }
        }
        return false;
    }
};

```

一直看不懂的巧妙解法：
```c++
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        return (s + s).find(s, 1) != s.size();
    }
};

```
# DP专题
## 动态规划

1. 整数划分

给定一个正整数 n ，将其拆分为 k 个 正整数 的和（ k >= 2 ），并使这些整数的乘积最大化。

状态转移方程：

$$
d[n]= max(  max(d[n-1]*1,(n-1)*1), max(d[n-2]*2,(n-2)*2),...,max(d[1]*(n-1),(n-1)*1))

$$

```python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        d=[0]*(n+1)
        d[1]=1
        for j in range(2,n+1):
            for i in range(1,j):
                d[j]=max(d[j], max(i*(j-i),d[i]*(j-i)))
        
        return d[-1]

```


数学方法

$$

数学规律：等分的时候乘积最大，重点是几等分。设x等分，则分成 n/x份，答案就是 x^{\frac{n}{x}}=x^{(\frac{1}{x})n},a^n是a的单增函数，则求max（a），即max(x^{\frac{1}{x}}) 


这个很简单，x=e的时候，最大，但是x需要是整数，比较2^{1/2}与3^{1/3},可知x=3时更大，即拆分的时候需要分成多个3时最大。

但是x可能不是3的倍数，可能余1，余2。余1的话，取max(3^{(n-1)//3},3^{(n-4)//3}*4),肯定是3^{(n-4)//3}*4更大。余2的话，取max(3^{(n-2)//3}*2,3^{(n-5)//3}*5),肯定是前者大。



$$
```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if n <= 3: return n - 1
        a, b = n // 3, n % 3
        if b == 0: return int(math.pow(3, a))
        if b == 1: return int(math.pow(3, a - 1) * 4)
        return int(math.pow(3, a) * 2)

```


2. 不同的二叉搜索树

```c++

class Solution {
public:
    int numTrees(int n) {
        vector<int> G(n + 1, 0);
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }
};


3. 编辑距离

```c++

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m=len(word1)
        n=len(word2)
        dp=list(range(n+1))
        for i in range(m):
            lu=dp[0]
            dp[0]=i+1
            for j in range(n):
                dp[j+1],lu=min(dp[j]+1,dp[j+1]+1,lu+int(word1[i]!=word2[j])),dp[j+1]
        return dp[-1]
```
## 普通DP
最小花费爬楼梯，这题楼梯顶部指的是数组end后一位，因此遍历完数组后，还要进行一次ans=min(a,b).

### 整数拆分，很有意思，结合指数e的理解看，就很直观

因为总数一定的情况下，拆成n份，并使其乘积最大，可以写成 （a/n）^n ,a>n, 实际上在问总量一定时，增长最大的指数底是多少，当然应该是自然数e，3是最接近的整数，因此就是3.

```c++
class Solution {
public:
    int integerBreak(int n) {

        int a=n/3,b=n%3;
        if(n<=3) return n-1;
        if(b==1) return pow(3,(a-1))*4;
        if(b==2) return pow(3,a)*2;
        return pow(3,a);
    }
};
```
### 不同的二叉搜素树: 总长度为i 的根节点为j 的BST的可能种数是 左边j-1个（最少为1个）元素组成的BST 数量 * 右边 n-j个 组成的数量


从1到n的n个自然数，组成的BST的种数

```c++
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """

        
        d=[0]*(n+1)
        d[0]=1
        d[1]=1
        
        for i in range(2,n+1):
            for j in range(1,i+1):
                d[i]+=d[j-1]*d[i-j]
        return d[n]
```
很好的一道dp题。

### 不同路径II：基础二维DP的升级版本，加了障碍，非背包问题的空间优化不用使第二维for 倒序
```c++
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """

        m=len(obstacleGrid)
        n=len(obstacleGrid[0])

        ans=[0]*n # 空间优化版，所以只用申请n个长度
        ans[0]=1 if obstacleGrid[0][0]!=1 else 0 # ans[0]的初始化 
        for i in range(1,n):# 先更新第一行的走法，便于后面使用
            if obstacleGrid[0][i]==1:
                break
            ans[i]=ans[i-1]
        for i in range(1,m):
            if obstacleGrid[i][0]==1:
                ans[0]=0
            for j in range(1,n):
                ans[j]=ans[j-1]*int(obstacleGrid[i][j-1]!=1)+ans[j]*int(obstacleGrid[i-1][j]!=1)
                #空间优化版，j-1是左侧走来的，j就是从上面走来的种数，但是要看能不能走通，不能走通就不需要加
                if obstacleGrid[i][j]==1: # 还得看这个点是不是障碍，是的话也走不通
                    ans[j]=0
        ##print(ans)
        return max(ans[-1],0) if obstacleGrid[-1][-1]!=1 else 0

```
也是很好的一个dp。空间优化版。



#TODO 

1. 监控二叉树不知道为什么不能AC—— AC了之前想错了，看新的
```C++  
//新的
class Solution{
public:
    int num=0;
    int minCameraCover(TreeNode *  root){
    
        if(recursive(root)==1){// 子节点的父节点一定要装摄像头，但如果根节点是子节点，没有父亲了，所以得在根节点上加一个摄像头
            num++;
        }
        return num;
    }

    int recursive(TreeNode *  root){
        if(!root) return 0;// 如果是空，则返回0
        int l=recursive(root->left);
        int r=recursive(root->right);
        if(r==0&&l==0) return 1;//告诉其父节点，这个节点是叶子节点
        else if(r==1||l==1) {num++;return 2;}
        else{
            // 否则一定是一个子节点装了摄像头，一个是空，或者两个子节点都装了摄像头，那么当前这个节点可以不考虑了,扔掉
            return 0;// 代表当前节点已经安装了摄像头
        }

    }
};public:
    int ans=0;
    void border(TreeNode* & root){
        if(root==nullptr)return;
        border(root->left);
        border(root->right);
        if(root->right||root->left){
            ans++;
            root=nullptr;
        }
    }
    int minCameraCover(TreeNode* root) {
        if(!root->left&&!root->right)return 1;
        border(root);
        //cout<<(root->val);
        return ans;
    }
};

```
### 最大子序和：简单题
给定一个整数数组 nums ，找到一个具有最大和的 __连续__ 子数组（子数组最少包含一个元素），返回其最大和
对比一下 贪心算法和 dp 算法。

这道题得重点在于连续，所以就算是遇到num[i]为负了，也得加，因为这个负数后面的连续序列得和没准就抵消掉这个负数得影响力。因此只要一个子序列的和不小于0，就继续向后加，并同时记录最大值。
贪心

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int result = INT32_MIN;
        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            count += nums[i];
            if (count > result) { // 取区间累计的最大值（相当于不断确定最大子序终止位置）
                result = count;
            }
            if (count <= 0) count = 0; // 相当于重置最大子序起始位置，因为负数部分没必要加到新的子数组里
        }
        return result;
    }
};
```

dp

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp=[0]*len(nums)
        dp[0]=nums[0]
        result=dp[0]
        for i in range(1,len(nums)):
            dp[i]=max(dp[i-1]+nums[i],nums[i])
            result=max(result,dp[i])

        return result
```

__总结__: 最普通的DP，就是找到问题是否符合 __DP规律__:_规模为n的最优问题一定是由规模为n-1的最优问题推导而来的_。

以最大连续子序和为例，含有10个元素数组的最大子序和，一定是前9个元素最大子序和 加/不加 第10个元素，因此符合DP规律。


## 背包问题 ： 一种特殊的2维遍历的DP问题
为什么是二维遍历？

一个容量为n的背包，其最大价值，一定可以从容量为n-j时的背包的最大价值 推导过来。 这里的j是指当前想要放进去的物体的体积，但是有n个物体，所以对于从n-j 状态 推导 到 n 状态
还需要一个for 遍历 n个物体。
```python
for i in objects:
    if weights[i]<=10:# 对于第i个物体，如果10>=Weight i,代表肯定能放下i，那就腾出Weights[i] 的空间
        dp[10]=max(dp[10],dp[10-weights[i]]+values[i])# 腾出空间和不腾出空间作比较
```
那么dp[10-weights[i]] 该如何获得？即不放第i个物体时的最大价值。

现在推导一下dp[i<10]的值应该是什么？
```python
dp[0]=0 # 容量为0时的最大价值肯定是0

dp[1]= for i in objects:
            if weights[i]<=1:
                max(dp[1],values[i]) # 容量为1时的最大价值就是可以放进去的所有values的最大值
# ps： 方便理解，我们假设物体最小重量为1                

dp[2]=for i in objects:
            if weights[i]<=2:
                max(dp[2],?)#容量为2的话， 没准可以放俩，也可以放一个，这时候用values[i]就不合适了
      # ? 应该写成什么。dp[1] 现在已经代表放一个时的最大值了， 那么如果能放下两个就放呗，则问号处应该写成 dp[2-weights[i]]+value[i]


# 同理
dp[3]=for i in objects:
        if weights[i]<=3:
            max(dp[3],dp[3-weights[i]]+values[i])
# dp[1]也可以改写成

dp[1]=for i in objects:
            if weights[i]<=1:
                max(dp[1],dp[1-weights[i]]+values[i])
```

如果我们把上述从dp[1]更新到dp[10]的过程写成循环的形式就是：

```python
dp[0]=0
for j in range(1,11):
    for i in objects:
        if weights[i]<=j:
            dp[j]=max(dp[j],dp[j-weights[i]]+values[i])
# 或者把 for i in objects 放到第一层，也是等价的
for i in objects:
    for j in range(1,11):
        if weights[i]<=j:
            dp[j]=max(dp[j],dp[j-weights[i]]+values[i])

```
但是上述两段代码都是不对的，为啥那，因为 __会算重__, 举例dp[3]更新的时候，考虑了第3个物体，但是dp[2]更新的时候也考虑了第3个物体，dp[3]可能会由dp[2]更新而来，所以这样就算了两次物体3，算重了。

所以，普通背包问题一般写成2维dp数组的形式，以防止 算重。那

即在dp[j]的前面加上一个维度写成dp[i][j],表示装下前i个物品且容量为j的背包的最大价值，这样更新dp[i][j]的时候考虑dp[i-1]的状态，就能保证i只在本次才会被算进内，dp[0~i-1]的都不会算第i的物体。。

即
```python
dp[i][j]=max(dp[i-1][j],dp[i-1][j-weights[i]]+values[i])
```

但是也可以优化为一维数组（节省空间），重点是 __倒序遍历__，为什么要倒序? 后面的例题再谈。



最后d[i][j]一定是我们要的答案。

![](https://code-thinking-1253855093.file.myqcloud.com/pics/20210117171307407.png)
__注意初始化问题__: 
    1. dp[0] 需要初始值，或者dp[i][0],dp[0][j]需要初始值
    2. 一般长度为n的问题，dp数组长度都会开成n+1，dp[0]为初始值

__注意遍历顺序问题__ ： 主要指两个for循环的嵌套顺序和内层循环的前后顺序.

> 上述例子叫普通背包，也叫01背包，即每个物品只有放和不放两种状态，一定要注意改成1维数组后，内层循环要倒序。

> 还有一种背包问题: 完全背包。每个物品无限个，因此改成1维数组后内层就不用倒序了。


> 背包问题两层循环的顺序就像前面的例子说的，可以随意调换，谁内谁外都可以，但是还有一种dp问题叫 组合总数问题，内外循环顺序就要严格控制了。



### 分隔等和子集：是否可以将这个数组分割成两个子集，使得两个子集的元素和相等

这题一定要注意，dp迭代和背包一样要是max，之前卡很久，在于纳闷为什么一定要求max，原因是，保证dp[i][j]每一项就算不能都达到j，但也要记录最接近j的值，比如 1 3 4 1 ,d[2][5]如果直接等于d[1][5-4],那会影响d[4][6]的结果。

每个物体就放和不放两种状态，最大的时间复杂度一定为 O(2^N)。

dp[j]代表容量为j 的背包能放下的价值，其肯定是根据小于其容量时的情况更新来的;对所有物体而言，每个物体都可以放进来，


```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target=sum(nums)
        if target%2==1:return False
        target/=2
        dp=[[0]*(int(target)+1) for _ in range(len(nums))] # 注意要是target + 1
        for i in range(1,len(nums)):
            for j in range(int(target)+1):
                if i==1 and j>=nums[0]:dp[0][j]=nums[0] # 进行初始化
                if j>=nums[i]:# 注意要有等号
                    dp[i][j]=max(dp[i-1][j-nums[i]]+nums[i],dp[i-1][j])
                else:# 总容量都不够放下这个物品，那么只能不放
                    dp[i][j]=dp[i-1][j]    
        #print(dp)
        if dp[-1][-1]==target:return True
        else: return False
```

修改成一维滚动数组形式。因为i行的状态只和i-1行有关，因此可以直接更新每一行，然后用一个代表一行的一位数组表示每次的状态，但是要倒序，因为从第一个开始修改，那么第一个代表的状态就不是本行之前的状态了， 后续状态更新的时候，就会用错，但是倒着来就没错了。

![](./img/dp%E4%B8%80%E7%BB%B4%E6%BB%9A%E5%8A%A8%E6%95%B0%E7%BB%84.png)


```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target=sum(nums)
        if target%2==1:return False
        target/=2
        dp=[0]*(int(target)+1) 
        for i in range(1,len(nums)):
            for j in range(int(target),-1,-1):
                if i==1 and j>=nums[0]:dp[j]=nums[0]
                if j>=nums[i]:
                    dp[j]=max(dp[j-nums[i]]+nums[i],dp[j])   
        #print(dp)
        if dp[-1]==target:return True
        else: return False
```


### 最后一块石头的重量II


有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；

如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。

最后，最多只会剩下一块石头。返回此石头最小的可能重量。如果没有石头剩下，就返回 0。


__这道题的核心在于想明白他和分割等和子集是一样的__。

正常思路是，从集合里面调两个最接近的，但是如果最接近的不相等，挑最接近的一定更好吗?。

假设有7,8,x. 7,8 代表最接近的，x代表任意数字。挑完7，8后剩余1，然后和x求差，结果为|x-1|，假设不挑7，8，调x，7，则剩|x-7|，8，结果为||x-7|-8|>=|x-7|-|8|>=x-7-8>=x-15,当x=14的时候，应该先碰掉14和7或者14和8.

从上述这个例子其实可以发现，应该是划分成两个集合，两个集合的和越接近，最后的结果一定越小。

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        target=sum(stones)
        dp=[0]*(target//2+1)
        for i in range(len(stones)):
            for j in range(target//2,-1,-1):
                if j>=stones[i]:
                    dp[j]=max(dp[j],dp[j-stones[i]]+stones[i])
        return target-dp[-1]-dp[-1]
```

### 目标和
给定一个数组，每个元素前可以是+号，或者-号，最后组成一个算式，使得该算式结果==target 的所有+,-号组合方式有多少种。



和前两题一样，重点是要分析出来。既然只有+，-1,设+号集合为l，-号为r。则有
$$
l-r=target\\
l+r=s\\
即存在l=(target+s)/2, s为数组和
$$

这样就变成了和前面两题一样的题，寻找元素和==target+s.


此外组合状态转移方程一般是. 

>dp[j]+=dp[j-nums[i]]

因为现在dp代表的不是总和，而是总和为j时的组合数量,
$$
总和为j的组合数量=\sum _{value }总和为dp[j-value]的数量
$$

__组合类dp问题初值很重要__: 
$$
设nums=[1]， 总和为1的组合数量dp[1]=\sum dp[0]
$$

如果dp[0]=0，那么dp[1]就只能是1，但是dp[0]代表总和为0的组合方案，这种组合方案应该有且只有一种，就是一个也不选。也可以理解成，对dp[j]而言，当j=value（某个数的值）时,只取value本身就是1种组合方式，即
$$dp[j]+=dp[j-value](此时j-value=0)，\\相当于dp[j]+=1$$
```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        #思路：分成两组，一组是都加正号，一组是都加负号，然后目标是正号组合之和-负号组合之和=target
        #0-i项组成目标和为 j 的组合种数设为dp[i][j]，一定等于dp[i-1][j-nums[i]]+dp[i-1][j]，
        # pos+neg=sum, pos-neg=target => pos=(sum+target)/2
        postarget=(sum(nums)+target)//2
        if postarget*2!=(sum(nums)+target):return 0 # 两种特判
        if target>sum(nums) or target<-sum(nums):return 0 
        dp=[0]*(postarget+1) # 和为0的组合数量肯定为1，dp[0]=1，其他的=0对于相加操作没有影响，也可以赋值为0 
        dp[0]=1# 很关键
        for i in range(len(nums)):
            for j in range(postarget,nums[i]-1,-1):# 注意如果j没有nums[i]大，那就没必要更新，肯定不用放nums[i]进去
                dp[j]+=dp[j-nums[i]]
        return dp[-1]
```

### 一零和
```bash
 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

示例 1：

输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3

输出：4

解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。 其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
```
#### __二刷__

第二次刷，相当于还是没刷，看到题还是有点懵。但是感觉很像普通背包，每个字符串都是一个物品，每个拿或不拿就是一种情况。之前的题是背包中的价值总和达到target，这个是背包中最多含有的0和1，之前是一个目标，现在是两个目标，那么用dp[i][j]来代表最多含有i个0和j个1应该可以。

接下来看dp数组每一项代表的含义，这道题的目标是最大子集的长度，即状态递推时要使用max。



状态转移方程：

$$
设当前遍历到的str含有a个0，b个1则，dp[i][j]就是含有当前这个字符串，或不含有当前这个字符串两种情况中取值最大的
dp[i][j]=max(dp[i][j],dp[i-a][j-b]+1)
$$

__需要警惕的是，dp[i][j]虽然是二维数组，但是实际相当于之前的一维数组，因此遍历的时候需要倒序。__




```python
for s in strs:
    a,b=0
    for c in s:a+=1 if c=='0' else b+1
    for i in range(m,a-1,-1):
        for j in range(n,b-1,-1):
            dp[i][j]=max(dp[i][j],dp[i-a][j-b]+1) 
```

然后就是初值问题，因为一开始dp都没有字符串，并且还有0个和0个1的字符串的数量肯定为0(这里没有空字符串)，所以dp[0][0]=0也没问题。

最后所有的代码如下:

```python
        dp=[[0]*(n+1) for _ in range(m+1)]
        for s in strs:
            a,b=0,0
            for c in s:
                if c=='0':a+=1
                else: b+=1
            for i in range(m,a-1,-1):
                for j in range(n,b-1,-1):
                    dp[i][j]=max(dp[i][j],dp[i-a][j-b]+1) 
        return dp[-1][-1]
```


#### __一刷__

按照代码随想录的思路，不要误解成多重背包
不同背包的区别
https://www.programmercarl.com/0474.%E4%B8%80%E5%92%8C%E9%9B%B6.html#%E6%80%9D%E8%B7%AF
![](https://code-thinking-1253855093.file.myqcloud.com/pics/20210117171307407-20230310132423205.png)



核心思路是，先看看规模可不可以减少，含有5个str的列表，是不是可以由4个str的列表推导过来，肯定是可以的。

dp[i]代表有i个str时满足要求的最大子集数，那么如果在加一个列表依然满足要求，那么dp[i+1]=d[i]+1,否则dp[i+1]=dp[i]——这个如果那么不用考虑，在下面的dp状态转移中包含了。

dp[i]满足要求的最大子集数有几个那？ 设dp[i][j][k]代表含有最多含有j个0,k个1的最大子集数，那么其可以由之前的状态推导出来：dp[i][j][k]= max(dp[i-1][j-num0(i)][k-num1(i)]+1,dp[i-1][j][k])

但是j，k其实可以看作一个维度，只不过这个维度时2维的罢了，因此可以按照滚动数组的思想，消去i这个维度，但是后面那个维度要倒序，这里就是j，k倒序。

则更新为：
dp[j][k]= max(dp[j-num0(i)][k-num1(i)]+1,dp[j][k])

__到这里没结束，dp永远要考虑初始化!：这里结果都是正数，且dp[0][0]=0即可，所以初始化全0就行__

```c++
class Solution {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> dp(m + 1, vector<int> (n + 1, 0)); // 默认初始化0
        for (string str : strs) { // 遍历物品
            int oneNum = 0, zeroNum = 0;
            for (char c : str) {  // 计算01的数量
                if (c == '0') zeroNum++;
                else oneNum++;
            }
            for (int i = m; i >= zeroNum; i--) { // 遍历背包容量且从后向前遍历！,且小于zeroNum部分直接忽略
                for (int j = n; j >= oneNum; j--) {
                    dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
                }
            }
        }
        return dp[m][n];
    }
};

```


## 完全背包：完全背包和01背包问题唯一不同的地方就是，每种物品有无限件

__01背包和完全背包代码的唯一不同就是体现在遍历顺序上__

直接上结论： 用1维数组，即滚动数组法，碰到无限数量的dp，内层直接正序即可。

为什么？ 因为之前是为了防止dp更新的时候算重复，之前的被更改了，现在就是可以算重，反正可以一直放，那就重呗。相当于dp一维数组，每行向后更新一次，就是在之前最大价值的基础上，只要能放下就增加一倍。

代码随想录的图也体现了这个规律：
![](https://code-thinking-1253855093.file.myqcloud.com/pics/20210729234011.png)

### 零钱和(零钱兑换II)二刷 : 其实这个也算组合类dp问题

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。

```python
dp=[0]*amount
dp[0]=1
for i in coins:
    for j in range(i,amount+1):
        dp[j]+=dp[j-i]
return dp[-1]

```


### 零钱和 ： 这个最重要的是dp[0]=1
```python
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        dp=[0]*(amount+1)
        dp[0]=1# 很重要
        for i in range(len(coins)):
            for j in range(coins[i],amount+1):
                dp[j]=dp[j]+dp[j-coins[i]] # 凡是需要自己加自己的，一定要dp[0]=1
        return dp[-1]
```
## 最少类DP问题
### 零钱兑换I ：零钱兑换 I求的是满足和的最少硬币数

__最少类dp问题有个细节__:
1. 状态递推要使用min，所以初值一般都设置为 最大值。

零钱兑换也有个细节：
内层循环不能写成 for j in range(coins[j],i), 因此要加一个if 保证数组不溢出。

此外，这题是最少硬币数，所以不是组合和排列个数问题，因此不需要考虑循环顺序。


```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp=[float('inf')]*(amount+1)
        dp[0]=0
        for i in range(1,amount+1):
            for j in range(len(coins)):
                if  i>=coins[j] :dp[i]=min(dp[i],dp[i-coins[j]]+1)
        return -1 if dp[-1]==float('inf') else dp[-1]
```

### 完全平方数：
给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

需要注意的都是，内层循环要是$i**0.55 +1, 要保证能取值取到 i**0.5$
```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp=[float('inf')]*(n+1)
        dp[0]=0
        for i in range(1,n+1):
            for j in range(1,int(i**0.5)+1):
                dp[i]=min(dp[i],dp[i-j*j]+1)
        return dp[-1] if dp[-1]!=float('inf') else 0
```
二刷一遍过，真爽。

## 组合总和类题 

### 二刷爬楼梯4

这个题要是按照背包的思路理解，很容易懵。背包是拿一个，台阶怎么拿啊，背包里的物体可以无限个，台阶怎么无限个那？所以这道题 不能把台阶看成物体，应该把把每次走的步数看成物品，
$$

走到第i个台阶的方法数=\sum dp[i-j]
$$

因为每次都可以从1~m 步中选一个，所以肯定是完全背包问题，即内层循环正向遍历。

但是如果站在dp的角度理解，这道题应该是外层循环是层数，内层是m，如果颠倒了就错了，如下。

```c++
// 以m=2，n=3 为例
class Solution {
public:
    int climbStairs(int n) {
        int m=2;
        vector<int> dp(n + 1, 1);
        dp[0] = 1;
        for (int j = 1; j <= m; j++)
         { 
            for (int i = 1; i <= n; i++){ 
                if (i - j >= 0) dp[i] += dp[i - j];
                cout<<i<<" : "<<dp[i]<<endl;
            }
        }
        return dp[n];
    }
};
输出：
i: dp[i]
1 : 1
2 : 1
3 : 1
1 : 1
2 : 2
3 : 2 // 错了，dp[3]应该=3

```

这是因为如果要是m放在外层的话，dp更新的时候只能按照1，2，3，...m 的顺序更新，即dp[3]在m=2的时候更新为 dp[3]=dp[3](每次只能走一步时走到3的方法数)+dp[3-2]=1+1， dp[3]的此次更新代表的含义是，如果一次能走两步了，那么dp[3]=dp[3-2] (回退两步时的方法数)+dp[3] (每次只能走一步时的走到3的方法数)，缺少了一种情况，即先走两步，在走1步，所以最后的答案dp[3]错了.

__这很好的引出了下一个问题:__ 求的是组合数？还是排列数？

如果是求组合数的话，内层要保证是 __遍历背包(或者叫目标和)__，外层遍历 __物品__。

如果是求排列数的话，内层要保证是遍历物品，外层遍历背包(或者叫目标和)。

这道题显然是 求排列数。举个例子，同样是走到第3层，[2,1]和[1,2] 是两种不同的走法。

所以一定要保证 内层循环是物品，在爬楼梯这道题里就是每次可以爬的数量。




### 爬楼梯4，完全背包版本：一个n阶台阶，每次可以上1~m层，一共有多少种上法。
```python
class Solution {
public:
    int climbStairs(int n) {
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) { // 遍历台阶（对应目标和）
            for (int j = 1; j <= m; j++) { // 遍历每次可以走的步数（对应物品）
                if (i - j >= 0) dp[i] += dp[i - j];
            }
        }
        return dp[n];
    }
};
```
### 单词拆分： 这道题真的好，完全背包+排列问题，并且字符串的dp数组处理，以及初值，全涵盖了
```
给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
```
容易想到，dp数组的长度应该是s的 (长度+1),d[i]=true/false，代表s中前i个字符组成的子串是否可以由wordict中的单词构成
$$ 
dp[i]每次更新的时候应该是 dp[i-size(word)] \& \& [word in wordDict],\\
即如果前i-size(value)个字符能组成，\\且word在字典里，那么i就可以组成。
$$

想到这，后面的就比较容易了。
接下来就是 __初值__ 和 __遍历顺序__.

dp[0] 代笔的是空字符。 肯定要等于true，因为如果找到了 $d[i]=dp[0] \&\& [ word in wordDict]，dp[0] $ 肯定要为true，才不影响dp[i]得到正确的值。

此外遍历顺序一定要是i在外，因为这是一个排列问题，显然 'a'+'b'和'b'+'a' 是不一样的。

这里我一开始做的时候，特别容易在 dp[i]的更新的时候懵逼，我总觉得这题不能从0到i的遍历，因为字符串不能是往右扩张的，可能会把word插在i的左侧，但是这么想是错的：i除了第一次从空字符串变为某个word，后面的更新肯定是一直向右扩张的，因为开头要和s的开头部分一直，所以左边不能变了。其次，因为内层循环每次都是从0开始的，因此如果可以重新更新i为一个新的i，那也是可以实现的，这也是为什么是排列问题的核心原因。


```c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> w(wordDict.begin(),wordDict.end());
        vector<bool> dp(s.size()+1,false);
        dp[0]=true;
        for(int i=1;i<=s.size();i++)
        for(int j=0;j<i;j++)
        {
            string substr=s.substr(j,i-j);
            if(w.find(substr)!=w.end()&&dp[j])dp[i]=true;
        }
        return dp[s.size()];
    }
};
```
### 组合总数IV： 遍历顺序，必须先遍历背包

如果求组合数就是外层for循环遍历物品，内层for遍历背包。

如果求排列数就是外层for遍历背包，内层for循环遍历物品。

如果把遍历nums（物品）放在外循环，遍历target的作为内循环的话，举一个例子：计算dp[4]的时候，结果集只有 {1,3} 这样的集合，不会有{3,1}这样的集合，因为nums遍历放在外层，3只能出现在1后面


```python
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        dp=[0]*(target+1)
        dp[0]=1
        for i in range(1,target+1):
            for j in range(len(nums)):
                if i>=nums[j]:
                    dp[i] += dp[i-nums[j]]
        return dp[-1]
 
 ```

 ### 初值问题：零钱兑换I
 这里dp[0]应该是0，因为要从之前的状态推出来，dp[0]如果是1就会出错。

 其他dp应该是最大值，否则状态转移求最小的时候就会被0覆盖。

 ```python
 class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp=[float('inf')]*(amount+1)
        dp[0]=0
        for i in coins:
            for j in range(i,amount+1):
                dp[j]=min(dp[j-i]+1,dp[j])

        return dp[-1] if dp[-1]!=float('inf') else -1
```

### 完全平方数： 注意要i*=i，以及第一层range 最后要+1保证能取到 int(n**0.5)

```python
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[float('inf')]*(n+1)
        dp[0]=0
        for i in range(1,int(n**0.5)+1):
            i*=i
            for j in range(i,n+1):
                dp[j]=min(dp[j],dp[j-i]+1)
        return dp[-1] 
```


### 单词拼接：隐含了排列，因为 abc 由 a+b+c构成，如果顺序乱了，那肯定不行，所以隐含排列，排列问题，背包一定要在外

```python
        dp=[False]*(len(s)+1)
        dp[0]=True
        for i in range(1,len(s)+1): # 背包在外的话，要从1开始
            for word in wordDict:
                if i>=len(word):# 并且防止数组越界，要有这个判断
                    dp[i]= (dp[i-len(word)] and word==s[i-len(word):i]) or dp[i]
        print(dp)
        return dp[-1]
```

## 多重背包：就是每个物品有有限个

可以当作01背包看待，把每个物品展开就行了，如代码随想录的图。

![](./img/%E5%A4%9A%E9%87%8D%E8%83%8C%E5%8C%85%E4%BB%A3%E7%A0%81%E9%9A%8F%E6%83%B3%E5%BD%95.png)

![](https://code-thinking-1253855093.file.myqcloud.com/pics/%E8%83%8C%E5%8C%85%E9%97%AE%E9%A2%981.jpeg)

### 打家劫舍二刷

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警

1. 确定dp[i]的含义： 偷前i个房间 最大的盗窃金额
2. 确定状态转移方程： $dp[i]= max{dp[i-2]+money[i] (偷当前这个房间,就一定不能偷前一家，所以是dp[i-2]) ,dp[i-1] (代表偷前i-1家的最大值)}$ ，这个地方一定要搞清楚，dp[i-1]的含义，就是代表偷前i-1家，而不偷第i家时的最大价值，至于偷不偷 i-1家无所谓。
3. 确定初始值：dp[0]肯定等于0，因为偷0个房间肯定最大金额为0.因为递推公式出现了i-2，所以要从2开始进行for遍历，因此i=1时的初值也要考虑，dp[1]=money[1]（不解释了）。
4. 确定遍历顺序。

> 已经开始能按照代码随想录dp五部曲的思路进行思考了，更有条理了。


此外还有一些细节：
5. 当需要给dp[1]赋为初值时，需要考虑输入nums的长度是否大于等于1.
6. dp[1]的初值应该是max(nums[1],nums[0])
7. 这道题dp数组没必要多开一个，这样就容易和nums的标号搞混，dp[0]也代表第一个房间的偷窃价值就够了。




### 打家劫舍系列

最简单的这道，第一次做的时候做错了，原因是因为没想明白递推关系

dp[i]=max(dp[i-2]+nums[i],dp[i-1])

这个递推公式没有错，dp[i]代表偷到第i家时的最大值，dp[i-2]一定不会偷第[i-1]家，所以一定要加上第i家才最大，dp[i-1]一定不会偷到第i家，至于dp[i-1]偷不偷第[i-1]家，无所谓，不偷的话，和dp[i-2]的值一样，dp[i-2]+nums[i]就已经包含这种情况了。


以及，忘了考虑nums 长度小于2的时候进行一次判断，直接输出。

```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp=[0]*len(nums)
        dp[0]=nums[0]
        if len(nums)<=1:return dp[-1]
        dp[1]=max(nums[0],nums[1])
        for i in range(2,len(nums)):
            dp[i]=max(dp[i-2]+nums[i],dp[i-1])
        return dp[-1]
```


### 打家劫舍II 二刷
第二次做竟然不不会了，看了下第一次做的，第一次自己做的都AC了，很难受，不刷了，脑子不行了，明天再整吧。

主要卡点就是 __环的处理__.  

要想明白环的存在只影响第一个和最后一个元素，即如果打劫了最后一家，如果有环的存在，那么一定不能打劫第一家，最后一家和第一家只能选一个。

```c++


```


### 打家劫舍II

自己做的版本，AC了，思路是用两个dp数组，一个是一定不打劫第一家的，另一个是正常的
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        dp=[0]*len(nums)
        dp[0]=nums[0]
        if(len(nums)<=1):return dp[-1]
        dp[1]=max(nums[1],nums[0])
        dp2=[0]*len(nums)
        dp2[0]=0
        dp2[1]=nums[1]

        for i in range(2,len(nums)):
            dp[i]=max(dp[i-2]+nums[i],dp[i-1])
            dp2[i]=max(dp2[i-2]+nums[i],dp2[i-1])
        #print(dp,dp2)
        if dp[len(nums)-1]!=dp[len(nums)-2]:# 代表偷了最后一家
            return max(dp2[-1],dp[-2]) #则返回不打劫第一家的情况和不打劫最后一家情况下的最大值
        return dp[-1]
```


### 打家劫舍III

正常暴力版本:会超时，但是思路得有，根据父节点是否抢劫来判断。

```c++
class Solution {
public:
    // unordered_map<TreeNode*,int> umap;
    int rob(TreeNode* root) {
        if (root == NULL) return 0;
        if (root->left == NULL && root->right == NULL) return root->val;
        //if (umap[root]) return umap[root]; 如果有直接返回
        // 偷父节点
        int val1 = root->val;
        if (root->left) val1 += rob(root->left->left) + rob(root->left->right); // 跳过root->left，相当于不考虑左孩子了
        if (root->right) val1 += rob(root->right->left) + rob(root->right->right); // 跳过root->right，相当于不考虑右孩子了
        // 不偷父节点
        int val2 = rob(root->left) + rob(root->right); // 考虑root的左右孩子
        //umap[root] = max(val1, val2); // umap记录一下结果
        return max(val1, val2);
    }
};
```

## 树形DP：这个偷东西的题一定要是后序遍历，因为要先遍历左右子树，才能算出偷不偷父节点对应的最大价值

### 二刷树形dp：其实关键的点是树的遍历如何和dp思想结合

对于这题来说，对于root来说，偷或者不偷，肯定是根据其子节点的状态推导而来的，借鉴前面的思路，$dp[root]=max(dp[root-1],dp[root-2]+root.val) (root-i)$代表当前节点下i层的节点，但是这样会发现，root-1有左右两个节点，因此dp数组应该长这样[(l,r),(l,r),(l,r),....]，

状态转移方程变为：
$dp[root]=max(max(dp[root-1][0],dp[root-1][1]),dp[root-2]+root.val)$
根据这个思路，root的状态需要由left和right推导得到，因此需要用后序遍历，我们得到两个子节点的最大值了，但是dp[root-2]怎么得到，后序遍历无法得到孙子节点的状态，我们可以尝试找到孙子状态和儿子状态的联系，用root-1代替root-2.

$dp[root-2]=dp[root-1](不偷root-1家)$，因此我们可以让dp数组记录一下是否偷第i家。

```python
class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def postorder(root):
            if not root : return (0,0)
            unrobl,robl=postorder(root.left)
            unrobr,robr=postorder(root.right)
            unrob=max(robl,unrobl)+max(robr,unrobr)
            rob=root.val+unrobl+unrobr
            return (unrob,rob)

        return max(postorder(root))
```

```C++

class Solution {
public:
    vector<int> postorder(TreeNode * root){
        if(!root) return vector<int>{0,0};
        vector<int> left=postorder(root->left);
        vector<int> right=postorder(root->right);
        // 偷父节点
        int val1=root->val+left[0]+right[0] ;//left [0] 代表左子树中不偷左子节点时的最大值
        // 不偷父节点
        int val2=max(left[0],left[1])+max(right[0],right[1]);
        return vector<int>{val2,val1};
    }
    int rob(TreeNode* root) {
        if(!root)return 0;
        vector<int> r=postorder(root);
        return max(r[0],r[1]);

    }
};
```
## 多重DP: 接下来的DP是多重DP，即每个dp时刻有两个及以上的状态，需要dp[i]增加一个维度,所以叫多重DP

## 买股票系列-DP方法
I是只能买一次。

对于普通买股票问题。

换个思路，想这么一个问题，一个股票大神不管操作时间多长，是不是总能赚的最多。

肯定有赚的最多的情况，那么就一定有这么一个大神（这个大神可以吃后悔药，即它可以修改自己的历史操作）。——因此一定可以dp做。

但是有状态转移方程吗？


dp[i]代表操作周期为i天时的最大收益。
但是如果这么定义，只有卖出了才有收益，那么dp[i]一定是已经卖出了。如果想等到dp[i+1]再卖出一定需要i之前不卖出时的收益，所以这么定义dp是不够的。

每个dp时刻要记录两个状态的信息：

dp[i][0]: 不卖的收益(不卖就是负收益)

dp[i][1]: 卖掉的收益 

dp[i+1]如何变化？

dp[i+1] 的选择就三种，卖出，买入，不动。

卖出（i+1之前也可以卖）：

dp[i+1][1]=max(dp[i][0]+day[i],dp[i][1])

买入：
dp[i+1][0]=max(dp[i][0],-day[i])

__其实就是拆分成两步，股票收益可以看作是买入收益+卖出收益，因为是同一只股票，最优情况肯定是买入收益最大（花最小成本）+卖出收益最大。所以开两个dp记录就行了，而且状态i的卖出收益一定与i-1的买入最大收益有关__


```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if (len == 0) return 0;
        vector<vector<int>> dp(len, vector<int>(2));
        dp[0][0] -= prices[0];// 注意初值一定要想对！！！！
        dp[0][1] = 0;
        for (int i = 1; i < len; i++) {
            dp[i][0] = max(dp[i - 1][0], -prices[i]);
            dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0]);
        }
        return dp[len - 1][1];
    }
};
```

还可以再精简，因为只与前一个时刻有关，可以化为滚动数组,但是dp[][1]的更新需要依赖dp[i-1]，所以需要两个dp[][],确保dp[i][1]更新的时候dp[i-1][0] 还是i-1之前的。

```c++
dp[i % 2][0] = max(dp[(i - 1) % 2][0], -prices[i]);
dp[i % 2][1] = max(dp[(i - 1) % 2][1], prices[i] + dp[(i - 1) % 2][0]);
```

### 买卖股票II：注意与贪心的不同

II是可以买多次，但每次买之前要卖掉之前的。

因为必须要先卖出去，才能买，所以dp[i][0]代表的持有收入必须要在当天卖掉之前的才能才买入，而且当天卖完就可以买，因此dp[i][0]还可以代表买卖完n次后再买的最大收益，只需要在-price加上卖出的收益dp[i-1][1]



> 这里有一个特别值得注意的地方，为什么第i天买入 要加上i-1天时的卖出收益，而不是i天的卖出收益，如果第i天股价更高，在第i天卖掉之前买的难道收益不更大吗？ 其实这么想没错，那么买入状态转移方程就要更新为：

```
dp[i ][0] = max(dp[(i - 1)][0], max(dp[i - 1][1],prices[i]+dp[i-1][0])-prices[i])

```
> 但是，上述状态方程可以化简， 当第二个max中，取prices[i]+dp[i-1][0]时，第一个max的第二项就为 prices[i]+dp[i-1][0]-prices[i]=dp[i-1][0]，和第一个max的第一项相等，因此第二个max的第二项就可以被优化掉。 从直观上理解，就是 __如果选择在第i天卖掉之前的，并且在第i天买入股票，则相当于在第i天没有买入，第i天的买入收益为 prices[i]-prices[i]=0, i天前的买入收益==i-1天前的买入收益__


然后加上，奇偶优化：
```c++
dp[i % 2][0] = max(dp[(i - 1) % 2][0], dp[(i - 1)%2][1]-prices[i]);
dp[i % 2][1] = max(dp[(i - 1) % 2][1], prices[i] + dp[(i - 1) % 2][0]);
```

### 买卖股票III：交易两次

之前说了买卖股票最大收益可以等同于求 max（-花费）+ max(卖出)，所以要同时维护两个dp数组，则写成了dp[i][0]代表max(-花费)，dp[i][1]代表最大卖出收益。

至多交易两次下的最大收益，也可以分解为： 在max（至多一次交易），max（至多两次交易）中求最大。

一次交易需要两个dp数组维护。 二次交易也需要两个dp数组维护，并且肯定是在一次交易的状态上进行的，因此一共需要4个数组来维护。

dp[i][1]: 至多一次交易对应的持有收益。= max(dp[i-1][1],-price[i])

dp[i][2]: 至多一次交易对应的卖出收益。=max(dp[i-1][2],dp[i-1][1]+price[i])

dp[i][3]:至多两次交易下的持有收益。=max(dp[i-1][2]-price[i],dp[i-1][3])

dp[i][4]: 至多两次交易下的卖出收益。=max(dp[i-1][4],dp[i-1][3]+price[i])

然后要注意 __初始化__。

卖出收益初始状态肯定为0.

持有收益，第0天的其实是数组的第一项，肯定是-price[0]

```c++
dp[0][1] = -prices[0];
dp[0][3] = -prices[0];
for (int i = 1; i < prices.size(); i++) {
    
    dp[i][1] = max(dp[i - 1][1],  - prices[i]);
    dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
    dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
    dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
}
```



### 二刷 买卖股票IV ： 出问题了，初值问题


```python
for i in range(k):dp[0][i][0]=-prices[0]

```
上面这个初始化很重要，他的意思是第一天，不管是第几次买入，收益肯定都是-prices[0].

因为 __可以在同一天买卖一支股票__, 第k次在第一天买入，就是卖掉了前k-1次的，对于第一天而言，不管买卖多少次，__卖__ 出收益都是0.所以，dp[0][i][0]=-prices[0]


其次，dp转移中第二个for 一般不会写成 for j in range(k),有dp[i][j-1]的存在，会访问越界。

所以一般都会写成 for j in range(1,K+1),j=1 代表第一天，而不是0，这样j-1越界的问题就可以不处理，否则，要判断if j<1 : 用 0 代替 dp[i][j-1].





### 买卖股票IV：买卖K次
有了买卖两次的启示，买卖k次无非就是再加几个dp数组来维护就好了。
并且使用三维数字来维护更方便
```c++
    vector<vector<vector<int>>> dp(price.size(),vector<vector<int>>(k+1,vector<int>(2,0));
    for(int i=1;i<=k;i++){
        dp[0][i][0]=-prices[0];
    }
    
    for (int i = 1; i < prices.size(); i++) {
        for (int j=1;j<=k;j++){
            dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j-1][1] - prices[i]); // max( 保持不持有，卖掉再持有)
            dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j][0] + prices[i]); // max(之前卖出，今天卖出)

        }
    
    }
```
### 买卖股票含冷冻期 2 刷： 
重点是想清楚，每次交易后，由于第二天无法买入，导致状态的增加。

dp[i][0]: i天前的 买入收益：max(i-1天前买入，max(-prices[i] + 第[i-1]天为冷冻期的卖出收益,-prices[i] + [i-2]天前就卖出的收益))  
dp[i][1]: i天前的 卖出收益：max(i-1天前卖出， prices[i] + [i-1]前天买入的收益) 

>这时候发现，出现一个新的状态：第[i-1]天为冷冻期的卖出收益

dp[i][2]: 第i天为冷冻期的卖出收益： = 第[i-1]天的卖出收益

>这时候又多了一个状态: 第[i-1]天的卖出收益  

dp[i][3]: 第i天的卖出收益 ： = [i-1]天前买入+prices[i]

写成代码：
```python
dp[i][0]=max(dp[i-1][0],max(dp[i-1][2]-prices[i],dp[i-2][1]-prices[i]))
dp[i][1]=max(dp[i-1][1],prices[i]+dp[i-1][0])
dp[i][2]=dp[i-1][3]
dp[i][3]=dp[i-1][0]+prices[i]

# 同时注意初值： dp[0][0]=-prices[0], dp[0][1]=0,dp[0][2]=0,dp[0][3]=0
# 但是因为出现了i-2，要处理一下 i-2=-1时的情况，第二天的时，由于前2天不可能卖出，所以此时【i-2】[1]=0

```

最后的代码如下：

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        dp=[[0]*4 for _ in range(len(prices))]
        dp[0][0]=-prices[0] 
        for i in range(1,len(prices)):
            dp[i][0]=max(dp[i-1][0],max(dp[i-1][2]-prices[i],dp[i-2][1]-prices[i] if i>1 else -prices[i]))
            dp[i][1]=max(dp[i-1][1],prices[i]+dp[i-1][0])
            dp[i][2]=dp[i-1][3]
            dp[i][3]=dp[i-1][0]+prices[i]

        return dp[-1][1]

```

__总结__: 上述代码是自己写的，也AC了，和1刷按照代码随想录写的版本有点不一样，这里解释一下，代码随想录为了处理 i-2的初值问题，选择直接将dp[i][1]的状态修改为 前i-1天卖出的收益，这样 dp[i][1]的状态更新就变成了： 
```python
dp[i][1]=max(dp[i-1][1],dp[i-1][3]) #前i-1 天卖出= max(前i-2天卖出了，第i-1天卖出了),
```
这样可以避免i-2的出现，也很巧妙。此外，二刷版本和一刷版本，dp[i][2]和dp[i][3]的定义正好相反，回顾复习的时候注意一下。


### 买卖股票含冷冻期：可交易无限次，卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)

__代码随想录中的思路有点绕，下面的是自己改的，更好理解一点__

这道题很好，可以加深对DP的理解，为什么k次买卖要有那么多dp数组来维护，原因是第i天的状态其实是需要根据i-1天的状态来推导的，但是每天都可能存在很多种状态，最简单的一次买卖就是，买和卖两种状态，所以需要两个dp数组维护，k次买卖，会出现2k种状态。因此，这题多了一个冷冻期，就是多了状态，具体多了那几个状态就是核心了。

首先写出不含冷冻期交易无限次的状态转移。

```c++
dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i]); //i天持有
dp[i][1]=max(dp[i-1][1],dp[i-1][0]+prices[i]); //i天卖出

```

还需要有i天冷冻时的购买收益(因为冷冻时无法卖出，所以一个就够了)，但是冷冻需要前一天的状态推导，因此还需要一个dp[i][]代表第i天卖出了时的收益。同时dp[i][1]为了不和在第i天卖出这个状态重合，就要修改成，保持卖出（即在前i-1天卖出，第i天不卖出）。

```
dp[i][0]=max(dp[i-1][0],max(dp[i - 1][3] - prices[i], dp[i - 1][1] - prices[i])); //i天持有= max(保持持有，前一天冷冻了才能买i天的，前一天卖出了）

dp[i][1]=max(dp[i-1][1],dp[i-1][2]); //前i-1 天卖出= max(前i-2天卖出了，第i-
1天卖出了),

dp[i][2]=dp[i-1][0]+prices[i]；// 第i天卖出的收益= 前一天持有+i天卖出

dp[i][3]=dp[i-1][2]；// i天冷冻了的购买收益


```

还有就是结果要从三个状态中找最大： 最后一天卖出了，最后一天之前卖出的收益
```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n == 0) return 0;
        vector<vector<int>> dp(n, vector<int>(4, 0));
        dp[0][0] -= prices[0]; // 持股票
        for (int i = 1; i < n; i++) {
            dp[i][0] = max(dp[i - 1][0], max(dp[i - 1][3] - prices[i], dp[i - 1][1] - prices[i]));
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][2]);
            dp[i][2] = dp[i - 1][0] + prices[i];
            dp[i][3] = dp[i - 1][2];
        }
        return  max(dp[n - 1][1], dp[n - 1][2]);
        //从 最后一天卖出了，最后一天之前卖出之中选择最大的
    }
};
```

### 买卖股票带手续费和无限次买卖股票，和之前买卖股票的区别就是: 每次卖出的时候要-去手续费
首先需要注意的是，除了卖出的时候需要减去手续费，买卖无限次和买卖一次仅有一个区别：买卖无限次就是每天买入的时候， 买入收益 -price[i] 要加上之前的卖出收益，代表买入前要先卖出。

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n=prices.size();
        vector<vector<int>> dp(n,vector<int>(2,0));
        dp[0][0]=-prices[0];
        for(int i=1;i<n;i++){
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i]);
            dp[i][1]=max(dp[i-1][1],dp[i-1][0]+prices[i]-fee);
        }
        return dp[n-1][1];
    }
};
```

## DP 各种最长序列问题
## 2 刷 最长递增子序列
设前i个数中，以0开始，j结尾(i>j)的最大递增子序列长度为dp[j],那么以i结尾的最大递增子序列长度为：
dp[i]= max(dp[i],dp[j]+1 if nums[i]>nums[j])

__注意__ :   
1.dp各项初值应该为1，
2. 最后的结果应该为max(dp)，因为最大长度的子串不一定是以最后一个数字为结尾的

### 最长递增子序列：找到数组nums中最长严格递增子序列的长度

状态转移方程的思路：

设dp[i]为以第i项结尾的最长序列长度，其等于dp[0]-dp[i-1] 中，结尾项小于nums[i]的最长长度+1 与 dp[i] 中的最大值
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n=len(nums)
        dp=[1]*n 
        result=1 # 初值设为1，以应付【0】的情况
        for i in range(1,n):
            for j in range(0,i):
                if nums[j]<nums[i]:
                    dp[i]=max(dp[i],dp[j]+1) 
            result=max(result,dp[i])# 结果应该求所以最大
        return result
```


### 2 刷 最长连续递增序列：
因为是连续的，dp[i]只能由前dp[i-1]更新而来
dp[i]=dp[i-1]+1 if nums[i]> nums[j] 

```python
        result=1
        dp=[1]*len(nums)
        for i in range(1,len(nums)):
            dp[i]=dp[i-1]+1 if nums[i]>nums[i-1] else dp[i]
            result=max(result,dp[i])
        return result
```
### 最长连续递增序列: 在最长递增的基础上要求必须是连续的序列

只需要把在状态转移的时候只看 dp[i-1]即可，因为要保证连续。所以这是一道简单题。
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n=len(nums)
        dp=[1]*n 
        result=1 # 初值设为1，以应付【0】的情况
        for i in range(1,n):
            j=i-1
            if nums[j]<nums[i]:
                dp[i]=max(dp[i],dp[j]+1) 
            result=max(result,dp[i])# 结果应该求所以最大
        return result
```
### 2 刷最长重复子序列：重复子序列和公共子序列的区别是 前者连续，后者可以不连续

__注意__:
1. dp[i][j]的含义在重复子序列，即允许连续时代表，以i，j结尾的两个字符串的重复长度，在公共子序列中代表i，j前的字符串的重复长度。
2. 初始化融入到for循环中的技巧：dp向后移动一个位置，然后全部初始化为0 ，两个for都从1开始遍历到最后。

3. 公共序列只比重复序列多了一个状态，即两个字符串末尾不相同时，公共长度要取 包含i不包含j，包含j不包含i两种情况中最长的那个，因为包含i，i可能会和j之前的相同，反之同理，所以要考虑这种情况。

### 最长重复子序列: 子序列默认连续（两个数组，那么就需要dp[i][j]）

想状态转移的时候，很容易想到 第一个数组的前i项和第二个数组的前j项中的最长是一个状态，因此应该用dp[i][j]代表一个状态，遍历的时候肯定就要两个for。

但是dp[i][j]是代表i，j之前的最长重复子序列，还是代表以 i，j项结尾的最长重复子序列那，因为要求连续，所以应该是代表以i，j结尾的才对，因为这样i，j由 i-1，j-1状态更新的时候，就肯定是连续的了。

状态转移： nums1[i]==nums[j]时，dp[i][j]=dp[i-1][j-1]+1,

但是这样的话，因为i-1，j-1的存在就要初始化以防止数组越界。

__注意，这里因为要连续，所以i，j要代表以i，j结尾的两个子序列。__

for j in ：
dp[0][j]= 1 if nums2[j]==nums1[0] else 0

for i in ：
dp[i][0]= 1 if nums1[i]==nums2[0] else 0

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        result=0
        dp=[[0]*len(nums2) for _ in range(len(nums1))]

        # 初始化
        for j in range(len(nums2)):
            dp[0][j]= 1 if nums2[j]==nums1[0] else 0
            result=max(result,dp[0][j])
        for i in range(len(nums1)):
            dp[i][0]= 1 if nums1[i]==nums2[0] else 0
            result=max(result,dp[i][0])
        # 状态转移
        for i in range(1,len(nums1)):
            for j in range(1,len(nums2)):
                if nums1[i]==nums2[j]:
                    dp[i][j]=dp[i-1][j-1]+1
                result=max(result,dp[i][j])
        return result
```
还要两个for 循环，好麻烦。
想办法融进状态更新的for循环。
```
dp[0][j]= 1 if nums2[j]==nums1[0] else 0

dp[i][0]= 1 if nums1[0]==nums2[j] else 0
```

首先初始化成全0的， else可以省了。
```
dp[0][j]= 1 if nums1[0]==nums2[j]
```
``` 1  ```  看看能不能写成 ```dp[i][j]=dp[i-1][j-1]+1``` 这种形式。

dp[0][j]就是 i=0 的时候 j 从 0遍历到j，在状态转移中就是，```dp[0][j]=dp[0-1][j-1]+1``` ，和 ```dp[0][j]= 1 ```就差一个```dp[0-1][j-1]```,只要让```dp[0-1][j-1]=0```就可以了，因为dp全初始化为0了，所以除了```dp[-1][-1]```以外，正好都是0。

dp[i][0] 也同理.

也就是把状态更新的for循环直接改成i，j从0开始。除此之外还需要解决dp[-1][-1]的问题，直接让dp数组右移一位即可。
如下，dp[1][1]代表数字1 第0项，数组2第0项，因此，对于nums的索引而言，就要是i-1，j-1.
```python
        for i in range(1,len(nums1)+1):
            for j in range(1,len(nums2)+1):
                if nums1[i-1]==nums2[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
                result=max(result,dp[i][j])
        return result
```






### 最长公共子序列: 子序列可以不连续(不好想不好想，卡了很久)

__注意，这里因为不要连续，所以i，j要代表以i，j 或者i，j 之前结尾的两个子序列。__

一定要注意，对于这种代表i及i以前的dp数组，更新dp[i][j]的时候，可以由之前的状态更新而来，但具体是由那几个状态更新而来就需要详细分析。

承袭最长重复子序列的思路，如果nums1[i]==nums2[j],dp[i][j]=dp[i-1][j-1]+1

如果nums1[i]!=nums2[j]，i固定时，dp[i][j]=max(dp[i][0~j]),因为dp[i][j]代表j之前最大的，因此max(dp[i][0~j])=max(dp[i][j],dp[i][j-1]), 同时，由于nums1[i]!=nums2[j]，所以dp[i][j]一定=dp[i][j-1],故 i固定时，dp[i][j]=
dp[i][j-1].

现在思考i增加时，当i增加一项时，由之前的推导，dp[i][j]=dp[i-1][j]。这时候dp[i][j]既可以取dp[i-1][j]也可以取
dp[i][j-1]，那到底取谁那？ 实际上这个问题就是在问 nums1[i]!=nums2[j]时，是把nums1[i]加进来，还是把nums2[j]加进来考虑公共子序列，因此取二者最大就好了。


然后再避免复杂的初始化，右移dp数组。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp=[[0]*(len(text2)+1) for _ in range(len(text1)+1)]
        for i in range(1,len(text1)+1):
            for j in range(1,len(text2)+1):
                if text1[i-1]==text2[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
                else:
                   dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
        return dp[-1][-1]
```

### 不相交的线： 此题和最长公共子序列一样，重点在于要想明白是一样的

直接连最长公共子序列，由于子序列在两个序列里顺序一致，因此一定不会相交，也即是最多的不相交的线。

这里给一下c++版本的代码以加强记忆。

```c++
class Solution {
public:
    int maxUncrossedLines(vector<int>& A, vector<int>& B) {
        vector<vector<int>> dp(A.size() + 1, vector<int>(B.size() + 1, 0));
        for (int i = 1; i <= A.size(); i++) {
            for (int j = 1; j <= B.size(); j++) {
                if (A[i - 1] == B[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[A.size()][B.size()];
    }
};
```



## 编辑距离专题

### 判断s是否是t的子序列

有了之前公共，重复子序列的题，这个就简单了。
然而这题如果不用 dp 其实更简单。

双指针法：
```python
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i=0
        for j in t:
            if i<len(s) and s[i]==j:
                i+=1
        return i>=len(s)
```

dp： 关键是状态转移方程
如果s[i],t[j]相等 ：d[i][j] = d[i-1][j-1]+1

如果不相等，那么就肯定不能算上 t[j]：d[i][j]=d[i][j-1]

```c++
class Solution {
public:
    bool isSubsequence(string s, string t) {
        vector<vector<int>> dp(s.size()+1,vector<int>(t.size()+1,0));
        for(int i=1;i<=s.size();i++)
        for(int j=1;j<=t.size();j++)
        if(s[i-1]==t[j-1])dp[i][j]=dp[i-1][j-1]+1;
        else dp[i][j]=dp[i][j-1];
        return dp[s.size()][t.size()]==s.size();
    }
};
```
### 2 刷 不同的子序列：一开始还是没想对，想的是判断是否是碰到了子序列t的最后一项，如果碰到了ans自加1，但是这样的话，就偏离题意了，



__这题很好，值得好好体会__

二刷的体会：
__1. 一般递推方程出现dp[] = dp[]+ dp[]这种形式，初值肯定不能都为0，因为如果都为0，最后加来加去的结果还是0，此时就要思考初值问题了，一般都是某些初值要设为1__

__2. 这题的状态递推很重要，具体看下面代码的解释__
```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        dp=[[0]*(len(t)+1) for _ in range(len(s)+1)]
        for i in range(len(s)+1): dp[i][0]=1 # 可以理解成 s中前i项字符中含有空字符的 个数
        for i in range(1,len(s)+1):
            for j in range(1,len(t)+1):
                if s[i-1]==t[j-1]:dp[i][j]=dp[i-1][j]+dp[i-1][j-1] # 重点是dp[i-1][j]的加入，代表前i-1个字符中含有t中前j个字符的个数，如果前i-1个就含有了j，那么就算第i个字符==t中第j个字符，都已经含有了，更得考虑进去
                else: dp[i][j]=dp[i-1][j] # 不然的话，s中前i个和s中前i-1个含有t中前j个的数量肯定是一样的
        return dp[-1][-1]
```
### 不同的子序列： s中可以出现几个 不同的 子序列，都为t

d[i][j]的意思是 0-i项 的s 中出现 0-j项 的个数，

当s[i]=t[j]时， 按理说是 d[i][j]=d[i-1][j-1]+1

但这是算公共长度的，比如 abcddd，abd，abcd 和 abd 的公共长度= abc ，ab 的公共长度+1，abcdd和abd的公共长度= abcd 和 ab的公共长度+1；

但是在这题中，我们要的是公共子序列个数，对于abcdd 和 abd 这种情况除了 abc_d 和 abd 有一个公共子序列以外，abcdd还可以是 abcd_和 abd 组成一个，因此

状态转移公式应为： d[i][j]=d[i-1][j-1]+dp[i-1][j]


但是这样的话，如果所有dp都初始化为0，加来加去，就全是0了。
因此要重新思考初始化。

固定i，dp[i][0]是什么？ dp[i][0]应该是i中含有空字符的个数，应该为1，这样
dp[i][1]= dp[i-1][j-1]时，就会变成1了

固定j,dp[0][j]是什么？ 空字符含有j的个数，肯定为0.


```c++
        vector<vector<uint64_t >> dp(s.size()+1,vector<uint64_t >(t.size()+1,0));
        for (long long i = 0; i <= s.size(); i++) dp[i][0] = 1;
        for(long long i=1;i<=s.size();i++)
        for(long long j=1;j<=t.size();j++)
        if(t[j-1]==s[i-1])dp[i][j]=dp[i-1][j-1]+dp[i-1][j];
        else dp[i][j]=dp[i-1][j];
        return dp[s.size()][t.size()];
    }
```

__注意，有时候dp数组里的数会很大， 及时数组长度不是特别大，因为dp中的数都是在之前的状态上累加的，所以可能会很大，如果long long 报错 overflow 的话，因为long long 的64 位是 正负32位，的而uint64_t 是纯整数的 64位，范围更大。__

### 两个字符的删除操作： 相同就不需要删，不相同，肯定要删一个，保留删后依然花费了最少次数的，即min(dp[i-1][j]+1,dp[i][j-1]+1),+1 是因为要删除一次

注意初始化：

对于dp[i][0],肯定要等于i，因为要删i次，才能为空字符。
对于dp[0][j],同理。dp[0][0]=0,两个空字符不需要删除。
```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        dp=[[0]*(len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(dp)):dp[i][0]=i
        for j in range(len(word2)+1):dp[0][j]=j
        for i in range(1,len(dp)):
            for j in range(1,len(word2)+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1)
        return dp[-1][-1]
```
### 编辑距离
给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符

删除一个字符

替换一个字符


```
if (word1[i - 1] == word2[j - 1])
    不操作
if (word1[i - 1] != word2[j - 1])
    增
    删
    换
```

删的情况前一天已经说过了，增和换怎么办，删和增一样， 

比如更新dp[i][j]时，想在i索引处插入一个使其和j相等，就是=dp[i-1][j]+1.
> 这里有时可能会有疑惑，在i处插入了一个，那么i位置原来的字符不就被挤到后面了，难道不会影响后面的字符判断吗—— 不会！，插入只是一个虚拟操作，并没有真正插入进去，没有改变i的位置。

如果i要换，把i换成和j一样的，那么就和i==j是是一样的，dp[i][j]=dp[i-1][j-1]

三种情况取最小即可。
```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n,m=len(word1),len(word2)
        d=[[0]*(m+1) for _ in range(n+1)]
        #print(d)
        if n==0: return m
        if m==0: return n
        for i in range(1,n+1):
            d[i][0]=i
        for j in range(1,m+1):
            d[0][j]=j

        for i in range(1,n+1):
            for j in range(1,m+1):
                s=1
                if word1[i-1]==word2[j-1]:    
                    s=0
                #print(min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+s))
                d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+s)

        return d[-1][-1]

```

## 回文子串

```
当s[i]与s[j]相等时，这就复杂一些了，有如下三种情况

情况一：下标i 与 j相同，同一个字符例如a，当然是回文子串
情况二：下标i 与 j相差为1，例如aa，也是回文子串
情况三：下标：i 与 j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串就看aba是不是回文就可以了，那么aba的区间就是 i+1 与 j-1区间，这个区间是不是回文就看dp[i + 1][j - 1]是否为true。
以上三种情况分析完了，那么递归公式如下：
```
```python
if (s[i] == s[j]) {
    if (j - i <= 1) { // 情况一 和 情况二
        result++;
        dp[i][j] = true;
    } else if (dp[i + 1][j - 1]) { // 情况三
        result++;
        dp[i][j] = true;
    }
}
```

由于 dp[i][j]是由 dp[i + 1][j - 1] 推导而来的，在dp二维数组中二者的关系是
![](https://code-thinking-1253855093.file.myqcloud.com/pics/20210121171032473-20230310132134822.jpg)

__所以一定要从左下角往左上遍历，即第一层for是倒叙遍历__
__还要注意第二层for 要从i开始，否则没意义，因为dp[i][j]代表的是从i到j的回文串__
同时初值应为false.

```python
r=0
dp=[[False]*len(s) for _ in range(len(s))]
for i in range(len(s)-1,-1,-1):
    for j in range(i,len(s)):
        if s[i]==s[j]:
            if j-i<=1:
                r+=1
                dp[i][j]=True
            else:
                if dp[i+1][j-1]:
                    r+=1
                    dp[i][j]=True
        else:
            dp[i][j]=False
return r
```

### 最长回文子串

```python
class Solution(object):
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp=[[0]*len(s) for _ in range(len(s))]
        for i in range(len(s)-1,-1,-1):
            for j in range(i,len(s)):
                if s[i]==s[j]:
                    if j-i==0:  # 代表了else j一定大于等于i+1，那么j-1 就不会越界
                        dp[i][j]=1
                    else: dp[i][j]=dp[i+1][j-1]+2
                else:
                    dp[i][j]=max(dp[i+1][j],dp[i][j-1])
        return dp[0][-1]
```
# 基础知识
## 基础知识继续学习备忘

python 字典几个不常用但有用的函数：

- get(key,default) : 返回 键值为key 的元素，如果没有返回default
- popitem(): 弹出的第一个元素的键值对

1. 运算符重载

[csdn](https://blog.csdn.net/m0_51940505/article/details/118274408?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-118274408-blog-124114746.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-118274408-blog-124114746.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=1)

[csdn](https://blog.csdn.net/qq_51667621/article/details/120710197)

2.  c++ 迭代器

https://blog.csdn.net/weixin_41368411/article/details/124222020

3.  lambda 表达式 

```c++
[](const vector<int>& a, const vector<int>& b){return a[0] < b[0];}
```

这里[]是探测器，相当于python里的lambda，（）里是输入值，括号后面大括号里是函数内容。
```c++

void testInnerFunc(){ 
    // 函数内声明具名函数 
    //auto innerFunc = [](int value)
    //{ cout << "inner named func " << value << endl; }; 
    //innerFunc(9999);  
    // 匿名函数的使用 
    //[](int value){ cout << "Anonymous func " << value << endl; }(8888);}
```
不仅如此，lambda表达式也可以具有捕获当前作用域变量的功能，格式如[capture_list]，假如我们有一个需求来对datas的变量求和：
```c++
int sum = 0;
std::for_each(datas.cbegin(), datas.cend(), [&sum](int element){ sum += element;});
```
与for each 函数结合使用

```c++
std::for_each(datas.cbegin(), datas.cend(), [](int element){ // TODO something;});

```

4. 优先队列的使用： 优先队列本质是一个堆
定义为：
```c++
 priority_queue<Type, Container, Functional>
```
> __container 默认是vector, 一般也用不到别的__  
Functional 有两个: __greater\<Type>,less\<Type>__ 分别是小到大排序，从大到小排序，即大顶堆和小顶堆，__默认大顶堆__

- top(): 访问队头元素
- empty(): 队列是否为空
- size():返回队列内元素个数
- push():插入元素到队尾 (并排序)
- emplace():原地构造一个元素并插入队列
- pop():弹出队头元素
- swap():交换内容
### 经典例题：滑动窗口最大值，用堆维护： 因为是大顶堆维护的，所以可能顺序会变，所以用pair 存储下一开始的索引，只要是小于等于i-k的就应该删，因为肯定是最大的在最前面，所以即使没被删掉的也不会影响答案的正确性
```c++
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        priority_queue<pair<int,int> > q;
        for(int i=0;i<k;i++){
            q.emplace(nums[i],i);
        } 
        vector<int> ans={q.top().first};
        for(int i=k;i<nums.size();i++){
            q.emplace(nums[i],i);
            while(q.top().second<=i-k)q.pop();
            ans.push_back(q.top().first);
        }
        return ans;
        }
```

``` c++

int a[10]; // 定义一个 可以存放10个int 元素的 数组
a[0]=1;// 因为数组的索引是0开始的，所以第一个数是a[0]
a[1]=1;// 第二个数是a[1]
for(int i=2;i<10;i++) a[i]=a[i-1]+a[i-2];// 按照 斐波那契数列的定义，第i个数等于前两个数之和，所以对于第i个元素，前一个数的索引是 i-1，前两个数的索引是 i-2 
for(int i=0;i<10;i++) cout<<a[i]<<" ";//输出该数组的每个元素

```

python 中的优先队列是 导入headpq包
- python 优先队列默认小顶堆，如果要用大顶，可以先给每个元素加负号
```python
a = [12,2,4,5,63,3,2]
heapq.heappush(a,123) # 除了用 heaify 方法进行初始化，不然的话必须要用heappush压进去的元素才会自动排序

b = heapq.heappop(a) # 弹出堆顶，即min


a = [12,2,4,5,63,3,2]
heapq.heapify(a)#参数必须是list，此函数将list变成堆，实时操作。从而能够在任何情况下使用堆的函数

heapq.heappushpop(heap, item) =  先 push 再 pop

heapq.heapreplace(heap, item) = 先pop 再push

heapq.nlargest(n, iterable,[ key])

#查询堆中的最大n个元素

heapq.nsmallest(n, iterable,[ key])

#查询堆中的最小n个元素


将多个堆合并
a = [2, 4, 6]
b = [1, 3, 5]
c = heapq.merge(a, b)
>>[1, 2, 3, 4, 5, 6]

## 为什么要有nsmallest 函数和nlargest 函数？？—— 因为heapq 只会把最小的元素放到队头，但剩下的元素是乱序的，因此才需要两个函数，并且指定排序的key


```
__滑动窗口最大值python heapq 解法__
``` python

    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """

        q=[[-nums[i],i] for i in range(k)]
        heapq.heapify(q)
        ans=[]
        ans.append(-q[0][0])
        for i in range(k,len(nums)):
            heapq.heappush(q,[-nums[i],i])
            
            while len(q)!=0 and q[0][1]<=i-k: heapq.heappop(q)
            ans.append(-q[0][0])
        return ans
```

## 优先map
### Top K 个出现最多次的数


```c++
class Solution{
public:
    vector<int> Topkofnums(vector<int> nums){
        priority_map<pair<int,int> > q;
        for(auto i :nums){

        }
    }

};
```

## dfs 搜索减枝
## KMP算法
https://leetcode.cn/problems/repeated-substring-pattern/solutions/386481/zhong-fu-de-zi-zi-fu-chuan-by-leetcode-solution/


## 优先队列的使用（priority_queue）
## 堆的作用：

堆是为了有排序优先级的队列场景。如 有很多请求的任务，需要给他们动态分配处理顺序，按照各自的优先级，如果用一般的队列或者数组，每当进来一个新的优先级的任务时，需要和队内或组内所有的元素比较，才能排到新的位置。但是堆这种结构就可以减少比对次数，因为 新来的元素只用和 n/2 的父节点比就可以了，时间复杂度降到了O(nlogn)。

分为大顶堆和小顶堆，大顶堆就是父节点一定要比子节点大，反之亦然。



底层实现是一个堆
- 常见方法：
```
top 访问队头元素
empty 队列是否为空
size 返回队列内元素个数
push 插入元素到队尾 (并排序)
emplace 原地构造一个元素并插入队列
pop 弹出队头元素
swap 交换内容
```

https://blog.csdn.net/weixin_52192405/article/details/124593027
### STL 堆的一般使用
```c++
//升序队列，小顶堆
priority_queue <int,vector<int>,greater<int> > q;
//降序队列，大顶堆
priority_queue <int,vector<int>,less<int> >q;
 
//greater和less是std实现的两个仿函数（就是使一个类的使用看上去像一个函数。其实现就是类中实现一个operator()，这个类就有了类似函数的行为，就是一个仿函数类了）
```

### STL 堆的pair使用

```c++
priority_queue<pair<int, int> > a;
```

### STL堆的自定义使用

```c++

 6 struct tmp1 //运算符重载<
 7 {
 8     int x;
 9     tmp1(int a) {x = a;}
10     bool operator<(const tmp1& a) const
11     {
12         return x < a.x; //大顶堆
13     }
14 };
15 
16 //方法2
17 struct tmp2 //重写仿函数
18 {
19     bool operator() (tmp1 a, tmp1 b) 
20     {
21         return a.x < b.x; //大顶堆
22     }
23 };

priority_queue<tmp1> d;
priority_queue<tmp1, vector<tmp1>, tmp2> f;
```
https://www.cnblogs.com/huashanqingzhu/p/11040390.html

### TopK 

https://programmercarl.com/0347.%E5%89%8DK%E4%B8%AA%E9%AB%98%E9%A2%91%E5%85%83%E7%B4%A0.html

### 滑动窗口最大值

https://leetcode.cn/problems/sliding-window-maximum/solutions/

##  层序遍历思路


##  树的迭代遍历
# 代码技巧

c++的最小值最大值：
INT32_MIN


注意，给int a 赋值 INT32_MIN，如果a*2了则会超出范围，所以要，（long long）2*a

python 的最大值赋值：

a=float('inf')

在class里定义sort函数的cmp函数式，需要写成static bool 类型，否则会报错。

vector初始化：可以给首地址和尾地址进行初始化
vector<vector<int>>(que.begin(), que.end())

使用for(x:vector)遍历vector时如果修改了某个元素的内容一定要用引用，如for(&x:vector)

stoi(str)可以自动去除前导0

to_string(int)可以把int，float，long，double等转为str

c++ string 可以当作个栈，因为有.back()方法可以访问到最后一个元素，也有pop()方法弹出最后一个元素。
如这题 __删除相邻重复元素__
```c++
    string removeDuplicates(string s) {
        string ans;
       
        for(char i:s){
            if(ans.empty()||ans.back()!=i) 
                ans.push_back(i); 
            else
            ans.pop_back();
        }
        return ans;

    }
```
(1<<k) 是 2 的k 次方， k<<1,是k*2



##  带重复元素的全排列


```python 

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # res用来存放结果
        if not nums: return []
        res = []
        used = [0] * len(nums)
        def backtracking(nums, used, path):
            # 终止条件
            if len(path) == len(nums):
                res.append(path.copy())
                return
            for i in range(len(nums)):
                if not used[i]:
                    if i>0 and nums[i] == nums[i-1] and not used[i-1]:
                        continue
                    used[i] = 1
                    path.append(nums[i])
                    backtracking(nums, used, path)
                    path.pop()
                    used[i] = 0
        
        # 记得给nums排序
        backtracking(sorted(nums),used,[])
        return res
```



## 二叉树搜索树的修剪和BST的公共父亲
```c++
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        while (root && (root->val < low || root->val > high)) {
            if (root->val < low) {
                root = root->right;
            } else {
                root = root->left;
            }
        }
        if (root == nullptr) {
            return nullptr;
        }
        for (auto node = root; node->left; ) {
            if (node->left->val < low) {
                node->left = node->left->right;
            } else {
                node = node->left;
            }
        }
        for (auto node = root; node->right; ) {
            if (node->right->val > high) {
                node->right = node->right->left;
            } else {
                node = node->right;
            }
        }
        return root;
    }
};


```
## 删除二叉搜索树中的某个节点
这道题最重要的是想明白所有的情况，第一步肯定是先找到要删除的节点。找到节点后要分类讨论:

1. 如果该节点没有儿子，那么直接使其为空就好了。
2. 如果该节点有左儿子，但没有右儿子，那么使其为左儿子，反之亦然。
3. 如果左右儿子都有，那么要找到右子树里最小的元素作为新的该节点。


分析完上述情况，还要记得把要删除的节点的父亲节点的指向修改了。也要分类讨论：

1. 该节点没有父节点。
2. 该节点的父节点，只有左儿子
3. 该节点的父节点只有 右儿子

```c++
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        TreeNode *cur=root;
        TreeNode* f=nullptr;
        while(cur){
            if(cur->val>key){
                f=cur;
                cur=cur->left;
            }
            else if(cur->val<key){
                f=cur;
                cur=cur->right;
            }
            else{
                if(!cur->left&&!cur->right) cur=nullptr;
                else if(!cur->left)cur=cur->right;
                else if(!cur->right) cur=cur->left;
                else{
                    TreeNode * minInRight=cur->right,*ff=cur;

                    if(!minInRight->left)ff->right=minInRight->right;
                    else{
                    while(minInRight->left){
                        ff=minInRight;
                        minInRight=minInRight->left;
                    }
                    ff->left=minInRight->right;
                    }
                    minInRight->left=cur->left;
                    minInRight->right=cur->right;
                    cur=minInRight;
                }
                if(!f) return cur;
                if(f->left&&f->left->val==key) f->left=cur;
                else f->right=cur;
                break;
                }
                
                
            }
        return root;
    }
};
```


## 一些不常见的函数，但是刷题的时候要用到

- stol(): string to long ,把一个string转为long  

- stoll(): string to long long

- to_string(): int/long long to string


- string.find():
- - 查找某一给定位置后的子串的位置
从字符串s 下标5开始，（包括5！！）查找字符串b ,返回b 在s 中的下标

- - position=s.find(“b”,5);


### 递增子序列快速去重

核心思想：
保证同一层不要遍历相同的，因此可以用unordered set 或者set 记录一下同一层遍历过的，遇到相同的就跳。

这种去重思路更清晰，而且也可以解决组合问题，子集问题。
此外，set没有数组好，因为
>使用set去重，不仅时间复杂度高了，空间复杂度也高了，组合，子集，排列问题的空间复杂度都是O(n)，但如果使用set去重，空间复杂度就变成了O(n^2)，因为每一层递归都有一个set集合，系统栈空间是n，每一个空间都有set集合。
那有同学可能疑惑 用used数组也是占用O(n)的空间啊？
used数组可是全局变量，每层与每层之间公用一个used数组，所以空间复杂度是O(n + n)，最终空间复杂度还是O(n)。


```c++
class Solution {
private:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(vector<int>& nums, int startIndex) {
        if (path.size() > 1) {
            result.push_back(path);
        }
        int used[201] = {0}; // 这里使用数组来进行去重操作，题目说数值范围[-100, 100]
        for (int i = startIndex; i < nums.size(); i++) {
            if ((!path.empty() && nums[i] < path.back())
                    || used[nums[i] + 100] == 1) {
                    continue;
            }
            used[nums[i] + 100] = 1; // 记录这个元素在本层用过了，本层后面不能再用了
            path.push_back(nums[i]);
            backtracking(nums, i + 1);
            path.pop_back();
        }
    }
public:
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        result.clear();
        path.clear();
        backtracking(nums, 0);
        return result;
    }
};
```
### 全排列去重版
```python
class Solution:
    def dfs(self,l,end,nums):
        if l==end-1:
            self.result.append(nums.copy())
        seen=set()
        for i in range(l,end):
            if nums[i] in seen:continue
            seen.add(nums[i])
            nums[i],nums[l]=nums[l],nums[i]
            self.dfs(l+1,end,nums)
            nums[i],nums[l]=nums[l],nums[i]

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        self.result=[]
        self.dfs(0,len(nums),nums)
        return self.result

```

各种c++ hash 容器的优劣

![](https://cdn.mathpix.com/snip/images/_ZxAAwPijE3wNzkRaFkgGfEMyoNZZgLooe8RyLZmaKk.original.fullsize.png)


# 贪心专题
### 分饼干
对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。


```c++
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int index = s.size() - 1; // 饼干数组的下标
        int result = 0;
        for (int i = g.size() - 1; i >= 0; i--) { // 遍历胃口 
            if (index >= 0 && s[index] >= g[i]) { // 遍历饼干 
                result++;
                index--;
            }
        }
        return result;
    }
};
```
### 最大子序列和

__想明白贪心的策略，这里是只要子序列不比0小，就可以一直加，一直加，计算比以前变小了，但是result不更改就是了，但是如果count比0小了，肯定要从新开始算子序列了，因为负数部分的子序列一定不能要__.
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int result = INT32_MIN;
        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            count += nums[i];
            if (count > result) { // 取区间累计的最大值（相当于不断确定最大子序终止位置）
                result = count;
            }
            if (count <= 0) count = 0; // 相当于重置最大子序起始位置，因为遇到负数一定是拉低总和
        }
        return result;
    }
};

```

### 买卖股票1234
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/submissions/429390255/

贪心可以做，搜索左区间最小的，一直计算差值，记录最大。
```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int low = INT_MAX;
        int result = 0;
        for (int i = 0; i < prices.size(); i++) {
            low = min(low, prices[i]);  // 取最左最小价格
            result = max(result, prices[i] - low); // 直接取最大区间利润
        }
        return result;
    }
};
```

### 买卖股票II

很好的思路，就是如果当前天比前一天价格高，那就赚这个插值，如果买不了了就先卖掉（因为可以在同一天出售，相当于前一天就卖掉）

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size()==1)return 0;
        int sum=0;
        for(int i=1;i<prices.size();i++){
            int tmp=prices[i]-prices[i-1];
            if(tmp>0)sum+=tmp;
        }
        return sum;
    }
};
```
### 跳跃游戏：很好的体现贪心思想的题，就是难想
i每次移动只能在cover的范围内移动，每移动一个元素，cover得到该元素数值（新的覆盖范围）的补充，让i继续移动下去。

而cover每次只取 max(该元素数值补充后的范围, cover本身范围)。
```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        if(nums.size()==1) return true;

        int cover=0;
        for(int i=0;i<=cover;i++){
            cover=max(i+nums[i],cover);
            if(cover>=nums.size()-1)return true;

        }
        return false;
    }
};
```
### 跳跃游戏2
这里最重要的是cover要和nextcover分开，nextcover是下一步要走的覆盖范围，每一次更新到最大值即可，cover是当前这一步可以走到的最远处。下面的代码做了精简，因为不用走到最后一个位置，走到倒数第二个位置，如果i==cover，那就要继续走一步，所以ans++，如果i！=cover，就代表cover一定会大于最后一个位置的索引（因为题干说了一定能走到），ans就不用加了。
```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        if(nums.size()==1)return 0;
        int cover=0,nextcover=0;
        int ans=0;
        for(int i=0;i<nums.size()-1;i++){
            nextcover=max(nextcover,i+nums[i]);
            if(i==cover){
                cover=nextcover;
                ans++;
            }
        }
        return ans;
    }
};
```

### k次取反后的最大数组和
https://www.programmercarl.com/1005.K%E6%AC%A1%E5%8F%96%E5%8F%8D%E5%90%8E%E6%9C%80%E5%A4%A7%E5%8C%96%E7%9A%84%E6%95%B0%E7%BB%84%E5%92%8C.html#%E6%80%9D%E8%B7%AF

这是一道基础题，但是要注意一些细节。

### 分发糖果
这题思路很重要，如果要是从左向右遍历一次，那么又要考虑左边的孩子，又要考虑右边的孩子，右边的孩子要依次考虑右边的影响，因此就不好考虑了，所以可以分两次遍历，第一次从左往右考虑，第二次从右往左考虑。

```c++
class Solution {
public:
    int candy(vector<int>& ratings) {
        vector<int> candyVec(ratings.size(), 1);//这里要注意vector的初始化顺序，第一个参数为长度，第二个为初值
        // 从前向后
        for (int i = 1; i < ratings.size(); i++) {
            if (ratings[i] > ratings[i - 1]) candyVec[i] = candyVec[i - 1] + 1;//注意这里一定要是前一个的数量+1
        }
        // 从后向前
        for (int i = ratings.size() - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] ) {
                candyVec[i] = max(candyVec[i], candyVec[i + 1] + 1);
            }
        }
        // 统计结果
        int result = 0;
        for (int i = 0; i < candyVec.size(); i++) result += candyVec[i];
        return result;
    }
};
```

### 加汽油
最不好想的是，如果从0-i的汽油剩余总和小于0，那么不能从0-i的任意一个地方出发。

想明白这点，关键在于，因为题干说肯定能走完一圈，那么一定不能是一开始的时候就导致剩余小于0 ,肯当是走着走着0-i之间的后半段一直在耗油，所以走不完，那么假设是从c点开始耗油的，从0-c之间出发还是从c-i之间出发一定都不行，前者相当于走了0-i，后者更甚，连攒油的机会都没了，因此一定要从i+1开始走，从新攒油。
```c++
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int curSum = 0;
        int totalSum = 0;
        int start = 0;
        for (int i = 0; i < gas.size(); i++) {
            curSum += gas[i] - cost[i];
            totalSum += gas[i] - cost[i];
            if (curSum < 0) {   // 当前累加rest[i]和 curSum一旦小于0
                start = i + 1;  // 起始位置更新为i+1
                curSum = 0;     // curSum从0开始
            }
        }
        if (totalSum < 0) return -1; // 说明怎么走都不可能跑一圈了
        return start;
    }
};
```


### 引爆最多数量的气球

```c++
class Solution {
public:
    static bool cmp(const vector<int> &a,const vector<int> &b){// 这里的const 也很关键，不加const会报错，不知道为啥，
        return a[0]<b[0];
    }
    int num=1;
    int findMinArrowShots(vector<vector<int>>& points) {
        sort(points.begin(),points.end(),cmp);
        for(int i=1;i<points.size();i++){
            if(points[i][0]<=points[i-1][1]){
                points[i][1]=min(points[i][1],points[i-1][1]);//这步很关键！
            }
            else{
                num++;
            }
        }
        return num;
    }
};
```

### 根据身高重建队列

贪心体现在，插入的时候从队头开始一个一个看就行了。
但这需要有前提，就是要先排序。

并且排序还得注意身高相同的情况，身高相同的，前面人数少的要在前面

比如
>[[9,0],[7,0],[1,9],[3,0],[2,7],[5,3],[6,0],[3,4],[6,2],[5,2]]

纯根据身高排

>[[9,0],[7,0],[6,0],[6,2],[5,3],[5,2],[3,0],[3,4],[2,7],[1,9]]

再插入
(先【5，3】)

>[[6,0],[7,0],[5,2],[6,2],[5,3],[9,0],[3,0],[3,4],[2,7],[1,9]]
>[[3,0],[6,0],[7,0],[5,2],[3,4],[6,2],[5,3],[9,0],[2,7],[1,9]]

（先【5，2】）

>[[6,0],[7,0],[5,2],[6,2],[5,3],[9,0],[3,0],[3,4],[2,7],[1,9]]
>[[3,0],[6,0],[7,0],[5,2],[3,4],[6,2],[5,3],[9,0],[2,7],[1,9]]

最终结果对比

>[[3,0],[6,0],[7,0],[5,2],[3,4],[6,2],[5,3],[2,7],[9,0],[1,9]]
>[[3,0],[6,0],[7,0],[5,2],[3,4],[5,3],[6,2],[2,7],[9,0],[1,9]]

这样[5,3],[6,2]的位置就是错的了，原因就是因为先插了[5,3],再插了[5,2]
先插[5,3],再插[5,2],【5，3】放到了[6,2]后面，【5，2】放到了【6，2】前面，
但是如果式先插【5，2】再插【5，3】相当于是【5，3】顶掉了【6，2】的位置。
```c++
class Solution {
public:
    // 身高从大到小排（身高相同k小的站前面）
    static bool cmp(const vector<int>& a, const vector<int>& b) {
        if (a[0] == b[0]) return a[1] < b[1];//这个必须要有
        return a[0] > b[0];
    }
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort (people.begin(), people.end(), cmp);
        list<vector<int>> que; // list底层是链表实现，插入效率比vector高的多
        for (int i = 0; i < people.size(); i++) {
            int position = people[i][1]; // 插入到下标为position的位置
            std::list<vector<int>>::iterator it = que.begin();
            while (position--) { // 寻找在插入位置
                it++;
            }
            que.insert(it, people[i]);
        }
        return vector<vector<int>>(que.begin(), que.end());
    }
};

```

### 三个区间题
一般都是要先排序的，我习惯左排序。
合并重叠区间，合并的肯定要是一对重叠区间end最大的，即更新答案数组区间的【1】。

无重叠区间，也一样，只不过要的是删除的区间，因此肯定删掉end最大的，留下的就是min了。


#### 合并重叠区间
```c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> result;
        if (intervals.size() == 0) return result; // 区间集合为空直接返回
        // 排序的参数使用了lambda表达式
        sort(intervals.begin(), intervals.end(), [](const vector<int>& a, const vector<int>& b){return a[0] < b[0];});

        // 第一个区间就可以放进结果集里，后面如果重叠，在result上直接合并
        result.push_back(intervals[0]); 

        for (int i = 1; i < intervals.size(); i++) {
            if (result.back()[1] >= intervals[i][0]) { // 发现重叠区间
                // 合并区间，只更新右边界就好，因为result.back()的左边界一定是最小值，因为我们按照左边界排序的
                result.back()[1] = max(result.back()[1], intervals[i][1]); 
            } else {
                result.push_back(intervals[i]); // 区间不重叠 
            }
        }
        return result;
    }
};
```
####无重叠区间

这道题体现了引用的优势，函数参数使用引用和使用形参一定是前者快，这道题如果cmp函数不使用引用直接就会TLE，但是使用引用就可以AC。
```c++
class Solution {
public:
    static bool cmp(vector<int>a,vector<int>b){
        return a[0]<b[0];
    }
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end(),cmp);
        int count=0;
        for(int i=1;i<intervals.size();i++){
            if(intervals[i][0]<intervals[i-1][1]){
                count++;
                intervals[i][1] = min(intervals[i - 1][1], intervals[i][1]);
// 这个min是必须的，体现了贪心策略：即对于每一对重叠区间，删除end最大的，因此留下来的就是 end 最小的。
            }
        }
        return count;
    }
};
```
#### 划分字母区间
这道题的 贪心策略体现在，对于每个字母而言，找到其出现的最后一个位置，就可以划分一个区间，但是要不要划分还有继续往后看，如果后面的一个字母最后一次出现的位置更远，那么这两个字母必须要划分到一个区间里，如果后面的更近，那么也要划分到一个区间里。那什么时候结束这一个区间那？找到右边界了，就代表结束了。

```c++
class Solution {
public:
    vector<int> partitionLabels(string S) {
        int hash[27] = {0}; // i为字符，hash[i]为字符出现的最后位置
        for (int i = 0; i < S.size(); i++) { // 统计每一个字符最后出现的位置
            hash[S[i] - 'a'] = i;
        }
        vector<int> result;
        int left = 0;
        int right = 0;
        for (int i = 0; i < S.size(); i++) {
            right = max(right, hash[S[i] - 'a']); // 找到字符出现的最远边界
            if (i == right) {
                result.push_back(right - left + 1);
                left = i + 1;
            }
        }
        return result;
    }
};
```
### 单调递增的数字

这个题的贪心策略体现在

要让改变后的数最大， 那肯定是减少的最少，那么肯定是* * 9，并且变为9的前一项要-1（因为减少了），此外，更重要的是，这个过程是重复的，即，* * 9如果 ** 还是前比后大，那么后一位又要变为9，即使隔了几个，也是连续的9，比如212887，变成21279后，21要变成19那么后面的只能都变成99，因为要满足递增的，9已经是最大的了。

```c++
class Solution {
public:
    int monotoneIncreasingDigits(int N) {
        string strNum = to_string(N);
        // flag用来标记赋值9从哪里开始
        // 设置为这个默认值，为了防止第二个for循环在flag没有被赋值的情况下执行
        int flag = strNum.size();
        for (int i = strNum.size() - 1; i > 0; i--) {
            if (strNum[i - 1] > strNum[i] ) {
                flag = i;
                strNum[i - 1]--;
            }
        }
        for (int i = flag; i < strNum.size(); i++) {
            strNum[i] = '9';
        }
        return stoi(strNum);
    }
};

```

### 监控二叉树

这道题的贪心策略体现在：

每次从叶子节点开始，给叶子节点的父节点安装摄像头，之后，新的叶子节点变为了安装摄像头的节点的父节点，依次类推，直到根节点。


__知识1__:如何从叶子节点开始遍历——后序遍历
__知识2__:遍历过程中如何跳过父节点


```c++

```

# 链表题
## 设计链表： 这个也卡了很久，涉及的知识有： delete 删除指针指向的空间，dummyhead，node class 创建
```c++
class Listnode{
public:
int val;
    Listnode* next;
    Listnode(int val=0,Listnode * next=nullptr){
        this->val=val;
        this->next=next;
    }
};

class MyLinkedList {
private:
    int size;
    Listnode * dummyhead;
public:
    MyLinkedList() {
        size=0;
        dummyhead=new Listnode(0);
    }
    
    int get(int index) {
        if(index<0||index>=size) return -1;
        Listnode * cur=dummyhead->next;
        while(index--){
            cur=cur->next;
        }

        return cur->val;
    }
    
    void addAtHead(int val) {
        addAtIndex(0,val);
    }
    
    void addAtTail(int val) {
        addAtIndex(size,val);
    }
    
    void addAtIndex(int index, int val) {
        if(index>size) return;
        if(index<0)index=0;
        Listnode * cur=dummyhead;
        while(index--){
            cur=cur->next;    
        }
        Listnode *n=new Listnode(val);
        n->next=cur->next;
        cur->next=n;
        size++;
    }
    
    void deleteAtIndex(int index) {
        if(index<0||index>=size) return;
        Listnode * cur=dummyhead;
        while(index--){
            cur=cur->next;
        }
        Listnode *tmp=cur->next; // 删除空间
        cur->next=cur->next->next;
        delete tmp;
        tmp=nullptr;// 防止tmp成为野指针
        size--;
    }
};

```
## 删除元素：这道简单的链表卡了很久,原因是 没有考虑最后的一项==val。


因为对于链表，如果new=head，相当于把以head开头的链赋给了new如果最后一个node 的val ==val，那么new.next指向的是最后一个以不等于val的node 开头的链表，这个链表后面还连着一个val=val的node那，所以就需要在单独判断一次new.next是不是==val的，如果等于得删掉。
```python
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        new=ListNode()
        dummy=new
        while head:
            if head.val!=val:
                new.next=head
                new=new.next
            head=head.next
        if new.next and new.next.val==val:
            new.next=None
        return dummy.next
```
或者直接遍历下一个，就不用最后再判断一次了。
```c++
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* dummyHead = new ListNode(0); // 设置一个虚拟头结点
        dummyHead->next = head; // 将虚拟头结点指向head，这样方面后面做删除操作
        ListNode* cur = dummyHead;
        while (cur->next != NULL) {
            if(cur->next->val == val) {
                cur->next = cur->next->next;
                
            } else {
                cur = cur->next;
            }
        }
        head = dummyHead->next;
        delete dummyHead;
        return head;
    }
```
链表一定要有一个思维：就是每次遍历，不是只遍历一个node，由于有链的存在，相当于每次遍历都同时遍历了两个node，如图中蓝色框住的部分。
![](./%E5%88%A0%E9%99%A4%E9%93%BE%E8%A1%A8.png)

所以一般链表题，都会有个dummy head，也叫哨兵或者虚拟头，即一个空的指针指向链表的head。这样就可以保证一次遍历两个node 的情况下，每个节点都会被遍历到。

# 开头结尾问题

## 删除倒数n个节点——双指针法：第一个指针先走n，第二个再走，第一个走到头，第二个就到倒数第n个了
```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        first = head
        second = dummy
        for i in range(n):
            first = first.next

        while first:
            first = first.next
            second = second.next
        
        second.next = second.next.next
        return dummy.next


```
# 双指针、快慢指针
## 链表相交——双指针
```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode *pA = headA, *pB = headB;
        while (pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;
        }
        return pA;
    }
};

```

## 环形链表II: 只要我跑的快，我一定能追上比我慢的人

环形链表解决的问题是，找到一个可能有环的链表的相交节点。

第一个问题：如何判断是否有环—— 使用两个快慢指针，如果两个指针都到头了(即都为null)，那一定是没环的：有环一定会相遇。

第二个问题：找到相交节点——有环快慢指针一定会相遇，相遇了后就要找相交节点了。

设起点为0，相交节点为a，那么慢指针走到a的时候，快指针一定走了2a了。

1. 如果环的长度小于a，那么快指针已经领先一圈了，如果从相交节点算，那么相当于快的领先慢的a-l后，开始同时跑；

2 .如果环的长度=a，那么正好相遇的时候就是交点；

3. 如果环的长度大于a，那么快的还在第一圈中，设环的长度为l，相当于快的领先慢的a后开始跑。

__两种解法__:
- 解方程：
设慢的走了x后二者相遇，上述三种情况下有:

1. 2x+a-l-x=n*l
2. 2x-x=n*l
3. 2x+a-x=n*l

上述三种情况其实可以合并为一种，即 2x+a-x=k*l, 那么x=k*l-a,k取1，取2，取3.....

k取1的时候，代表第一次相遇，l-x=a，即x在环内距离相交点a处。

至此我们得到了一个关键的结论： 第一次相遇时距离相交点的距离和head距离相交点的距离一致！！
那么第二个问题就好求解了，当slow==fast时，让head和二者任意一个同时走，再相遇的时候就是相交节点！！。

2. 第二个方法是我的同门想出来的，更加直观，不用算，非常聪明！

上述情形相当于两个人跑步，开始时，快的领先慢的距离为a，慢的速度为快的两倍，问二者在哪里第一次相遇？

假设二者同时出发，那么二者每次相遇一定都在起点。

现在在加一个慢的，但是和快的在一起，即都在a处，那么第二个慢的一定在a处和快的相遇，此时第一个慢的正好在原点处。快的一定是超过了第一个慢的之后再追上的第二个慢的，那当快的遇到第一个慢的后，走的距离一定等于两个慢的之间距离的两倍(两个慢的之间的距离为a)，从第二个慢的位置往回减去2a处即为快的和第一个慢的相遇的地方，即在-a处相遇。

这样得到的结论和第一种方法一致，但更直观，更巧妙！！
```c++

class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
            // 快慢指针相遇，此时从head 和 相遇点，同时查找直至相遇
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    fast = fast->next;
                    slow = slow->next;
                }
                return slow; // 返回环的入口
            }
        }
        return NULL;
    }
};
```

# 两数，三数，四数，五数之和

几数之和 主要是锻炼各种stl map 模板的使用，双指针思想，排序，去重思想，很综合，很好，很重要！！

### 两数之和： map的使用
```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> hash;
        if (nums.size()==0) return vector<int>{};
        for(int i=0;i<nums.size();i++){
            auto other=hash.find(target-nums[i]);

            if(other!=hash.end()){
                return  vector<int>{i,other->second};
            }
            hash[nums[i]]=i;
        }

        return {};
    }   
};
```

### 三数之和：三个数的和为targt，重点在于去重

用map做，如果不考虑去重就是在两数之和的基础上，多加一个循环，对列表里的每一个元素，取target-nums[i], 作为剩下元素中两数之和的target。

```c++
class Solution{
public:
    vector<vector<int>> threeSum(vector<int>& nums){
        int target=0;
        vector<vector<int>> result;
        for(int i=0;i<nums.size();i++){
            unordered_map<int,int> map;
            for(int j=i+1;j<nums.size();j++){
                int newtarget=target-nums[i];
                auto re=map.find(newtarget-nums[j]);
                if(re!=map.end())result.push_back(vector<int>{nums[i],nums[j],newtarget-nums[j]});
                map[nums[j]]=j;
                
            }
        }
        return result;
    }
};
```
并且可以不用map，因为不需要知道索引，用set 就可以了。
```
unordered_set<int> set;
set.erase(c); 删除c。
set.insert(c); 插入c。
```
```
输入：nums=[-1,0,1,2,-1,-4]

输出：[[-1,1,0],[-1,-1,2],[0,-1,1]]
```
有重复的。

__借助有序排列去重__:

如果对于有序数组，这样重复的答案，只用看对应位是否重复就可以了。

此外，借助set可以去重，比如两数之和，遍历的时候把见过的放到set里，即使数组里有重复的，也只会记录一次。假设三个数分别是a，b，c,那么对于c的去重，只要在每次在set里找到c后，就把c从set中删除并且保证a，b不会和之前相同就可以了。

如何保证a,b不相同？排序后保证b不重复出现比较简单，即b[j]!=b[j-1].
同理保证a不重复出现,即 a[i]!=a[i-1].

但是这样会有问题,如果全是0那，如[0,0,0]，如果保证b[j]!=b[j-1]，那么，最后一个0会被跳掉。
所以得允许两个一样的出现，这样保证b，c可以有两个一样的，但是不能连着三个出现，那么一定会重复。

比如[1,0,0,0,0],第一次连续出现0，即遍历到1，0，0，我们允许。但是如果出现了[1,0,0,0]那肯定重了，[0,0]这种情况出现过了。
```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        // 找出a + b + c = 0
        // a = nums[i], b = nums[j], c = -(a + b)
        for (int i = 0; i < nums.size(); i++) {
            // 排序之后如果第一个元素已经大于零，那么不可能凑成三元组
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) { //三元组元素a去重
                continue;
            }
            unordered_set<int> set;
            for (int j = i + 1; j < nums.size(); j++) {
                if (j > i + 2
                        && nums[j] == nums[j-1]
                        && nums[j-1] == nums[j-2]) { // 三元组元素b去重
                    continue;
                }
                int c = 0 - (nums[i] + nums[j]);
                if (set.find(c) != set.end()) {
                    result.push_back({nums[i], nums[j], c});
                    set.erase(c);// 三元组元素c去重
                } else {
                    set.insert(nums[j]);
                }
            }
        }
        return result;
    }
};
```
### 三数之和-双指针法
既然排序了，那么重复的肯定在一起，那么我找到一组答案后，把重复的直接过掉不就去重了吗，但是这是三数之和，要同时保证a,b,c不重复出现，在保证a不重复的前提下，bc都要跳过相同的，因此需要两个指针。

```c++
class Solution{
public:
    vector<vector<int>>  threeSum(vector<int>& nums){
        vector<vector<int>>  result;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++>)
        {
            if(i==0&&nums[i]>0) return result;
            if(i>0&&a[i-1]==a[i])continue;
            int left=i+1;
            int right=nums.size()-1;
            while(left<right){
                if(nums[i]+nums[left]+nums[right]>0)right--;
                else if(nums[i]+nums[left]+nums[right]<0)left++;
                else{
                    while(left<right&&nums[left]==nums[left+1])left++;
                    while(left<right&&nums[right]==nums[right-1])right--;
                    right--;
                    left++;
                }
            }

        }
        return result;
    }
}
```

### 四数之和
给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组

三数之和 套一个for循环。
```c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> result;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++){
            if(nums[i]>target&&nums[i]>=0)break;
            if(i>0&&nums[i]==nums[i-1])continue;
            for(int j=i+1;j<nums.size()-1;j++){
                if(nums[i]+nums[j]>target&&nums[i]+nums[j]>=0)break;
                if(j>i+1&&nums[j]==nums[j-1])continue;
                int l=j+1,r=nums.size()-1;
                while(l<r){
                long s=(long)nums[i]+nums[j]+nums[l]+nums[r];
                if(s==target){
                    result.push_back(vector<int>{nums[i],nums[j],nums[l],nums[r]});
                    while(l<r&&nums[l+1]==nums[l])l++;
                    while(l<r&&nums[r-1]==nums[r])r--;
                    l++;
                    r--;
                }
                else if (s>target) r--;
                else l++;
                }
            }
        }
        return result;
    }
};
```

### 四数之和: 卡的点在于减枝，是target>0时，nums[i]>target才跳，比如-5，-4，-3，1 但是target是-11,-5+-4>target 就跳了的话是不对的，也就是target<=0时无法减枝
```python

def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
    result=[]
    nums.sort()
    print(nums)
    for i in range(len(nums)):
        if nums[i]>target and target>0:break
        if i>0 and nums[i]==nums[i-1]:continue
        for j in range(i+1,len(nums)):
            if nums[i]+nums[j]>target and target>0:break
            if j>i+1 and nums[j]==nums[j-1]:continue
            new_target=target-nums[i]-nums[j]
            left=j+1
            right=len(nums)-1
            while left<right:
                if nums[left]+nums[right]==new_target:
                    result.append([nums[i],nums[j],nums[left],nums[right]])
                    while left<right and nums[left]==nums[left+1]:left+=1
                    while left<right and nums[right]==nums[right-1]:right-=1
                    left+=1
                    right-=1
                elif nums[left]+nums[right]<new_target:left+=1
                else:right-=1
    return result
```
### 两数之和的升级版——三、四、五数相加：3，4，5数相加指的是从3，4，5个数组中各找一个使得和为0，不用考虑重复情况----分成两组，4数就分成2+2，3数就分成1+2，5数就分成3+2或2+3
``` python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        map={}
        for i in nums1:
            for j in nums2:
                map[i+j] = map.get(i+j, 0) + 1
        result=0
        for i in nums3:
            for j in nums4:
                    result+=map.get(0-i-j,0)
        return result
```
### 快乐数： 用unordered_set 判断是否重复出现过
```c++
    int c(int n){
        if(n==0) return 0;
        return (n%10)*(n%10)+c(n/10);
    }
    bool isHappy(int n) {
        unordered_set<int> s;
        int t=n;
        while(true){
            t=c(t);
            if(t==1) return true;
            if(s.find(t)!=s.end()) return false;
            s.insert(t);
        }
    }
```
# 二叉树与递归
## 二叉树的最大深度

### 递归解法

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        int l=maxDepth(root->left)+1;
        int r=maxDepth(root->right)+1;
        return max(l,r);
    }
};
```

### 层序遍历解法
```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        queue<TreeNode*> tree;
        tree.push(root);
        int d=0;
        while(!tree.empty()){
            int n=tree.size();
            while(n--){
                if(tree.front()->left)tree.push(tree.front()->left);
                if(tree.front()->right)tree.push(tree.front()->right);
                tree.pop();
            }
            d++;
        }
        return d;

    }
};
```

## 二叉树最小深度：根节点不算叶子节点，所以输出时要特判

### 递归解法：三刷时自己写的
```python
def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        l= self.minDepth(root.left)
        r= self.minDepth(root.right)
        if l==0 and r==0 : return 1
        return min(l if l !=0 else float('inf'),r if r != 0 else float('inf'))+1 
```

### 一刷解法：更好，更简洁
```c++
        if(!root) return 0;
        
        int l=minDepth(root->left);
        int r=minDepth(root->right);
        if(!root->left||!root->right) return l+r+1; 
        return min(l,r)+1;
```
## 对称二叉树
### 层序遍历解法

### 递归解法：
```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root: return True
        def recursive(a,b):
            if a==None and b==None:
                return True
            if a==None or b==None:
                return False
            return a.val==b.val and recursive(a.left,b.right) and recursive(a.right,b.left)
        return recursive(root.left,root.right)
```

## 翻转二叉树

```python
class Solution:
    def invertTree(self,root):
        if not root:return root
        root.right,root.left=root.left,root.right
        self.invertTree(root.right)
        self.invertTree(root.left)
        return root

```
## 二叉树的最近公共祖先
```python
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        def recursive(r):
            if not r: return None
            if r.val==p.val or r.val==q.val: return r
            l=recursive(r.left)
            w=recursive(r.right)
            if not l and not w:
                return None
            if not l:
                return w
            if not w:
                return l
            return r
        
        return  recursive(root)
```
## BST 公共祖先

```c++
class Solution{
public:
    TreeNode * lowestCommonAncestor(TreeNode * root,TreeNode* p,TreeNode *q ){
        if(!root) return root;
        if(root->val==q->val||root->val==p->val)return root;// 这个可以并入到else 里
        if(root->val>q->val && root->val>p->val) return lowestCommonAncestor(root->left,p,q);
        if(root->val<q->val && root->val<p->val) return lowestCommonAncestor(root->right,p,q);
        else return root;
    }
};
```


## BST 插入: 对于BST来说，插入更简单，因为BST的插入一定是创造新的叶子节点，不用改变原有的树的结构！！！
```c++
class Solution{
public:
    TreeNode* insertIntoBST(TreeNode* root,int val){
        if(!root){
            TreeNode * n= new TreeNode(val);
            return n;
        }
        if(root->val<val)root->right=insertIntoBST(root->right,val);
        if(root->val>val)root->left=insertIntoBST(root->left,val);
        return root;
    }
};
```
## BST中的众数： 由于BST中序遍历一定是从小到大的，相同的数一定是连续出现的
```c++
 // BST 的中序遍历是有序的，相同的数肯定是连在一起的，因此中序遍历一次，记录下出现最多次的数即可
class Solution{
public:
    vector<int>ans;
    int last=0;
    int m,max=0;
    void inorder(TreeNode* root){
        if(!root) return;
        inorder(root->left);
        int cur=root->val;
        if(cur==last)m++;
        else{last=cur;m=1;}
        if(m==max)ans.push_back(cur);
        if(m>max){max=m;ans=vector<int>{cur};}
        inorder(root->right);
    }
    vector<int> findMode(TreeNode * root){
        inorder(root);
        return ans;
    }
};

```
## 普通二叉树中的众数：就是遍历一遍然后map存下每个val出现的次数，最后sort，主要是锻炼map的使用，sort的使用

__有的同学可能可以想直接对map中的value排序，还真做不到，C++中如果使用std::map或者std::multimap可以对key排序，但不能对value排序，所以需要将map转为vector__

> vector<pair<int, int>> vec(map.begin(), map.end());

```c++
class Solution{
public:
    unordered_map<int,int> map;// map second初始值=0
    void forwardorder(TreeNode * root,unordered_map<int,int> map){
        if(!root)return;
        map[root->val]++;
        forwardorder(root->left);
        forwardorder(root->right);
    }
    bool static cmp(const pair<int,int>a,const pair<int,int>b){
        return a.second>b.second;
    }
    vector<int> findMode(TreeNode* root){
        vector<int> ans;
        if (root == NULL) return result;  //这两句当作固定的模板，要考虑空的情况！

        forwardorder(root);
        vector<pair<int,int>> vec(map.begin(),map.end());
        sort(vec.begin(),vec.end(),cmp);
        
        ans.push_back(vec[0].first);
        for(int i=1;i<vec.size();i++){
            if(vec[i].second==vec[0].second)ans.push_back(vec[i].first);
            else break;
        }
        return ans;
    }
};

```
## 合并二叉树：root1对应的root2 为空的话，直接把root1返回就可以，同理，root1为空直接把root2返回即可，二至都不为空，就相加

```c++
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        if(!root2)return root1; 
        if(!root1)return root2；//
        //root1= new TreeNode(root2==nullptr?0:root2->val);// 不用new， 直接把root2赋值过来就可以了
        root1->val+=root2->val;
        root1->left=mergeTrees(root1->left,root2->left);
        root1->right=mergeTrees(root1->right,root2->right);
        return root1;

    }
};
```

## 删除 BST中的节点：删除是最难的，要想清楚可能的情况，有五种

第一种情况：没找到删除的节点，遍历到空节点直接返回了
找到删除的节点
第二种情况：左右孩子都为空（叶子节点），直接删除节点， 返回NULL为根节点
第三种情况：删除节点的左孩子为空，右孩子不为空，删除节点，右孩子补位，返回右孩子为根节点
第四种情况：删除节点的右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
第五种情况：左右孩子节点都不为空，则将删除节点的左子树头结点（左孩子）放到删除节点的右子树的最左面节点的左孩子上，返回删除节点右孩子为新的根节点

```c++
class Solution(){
public:
    TreeNode* deleteNode(TreeNode* root,int val){
        if(!root)return root;
        if(root->val==val){
        if(!root->left){
            TreeNode* n=root->right;
            delete root;
            return n;// 左子树空，右子树不空也返回右子树，空了就返回空
        }
        else if(!root->right){
            TreeNode* n=root->left;
            delete root;
            return n;//左子树不空，右子树空
        }
        else{
            TreeNode* cur=root->right;
            while(cur->left){
                cur=cur->left;
            }
            TreeNode* n=root->right;
            cur->left=root->left;
            delete root;
            return n;
        }//都不空
        }
        else if(root->val>val)root->left= deleteBST(root->left,val);
        else root->right=deleteBST(root->right,val);
        return root;
    }
};

```

## 普通二叉树的删除，不是很好想，暂时没想很明白
```c++
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) return root;
        if (root->val == key) {
            if (root->right == nullptr) { // 这里第二次操作目标值：最终删除的作用
                return root->left;
            }
            TreeNode *cur = root->right;
            while (cur->left) {
                cur = cur->left;
            }
            swap(root->val, cur->val); // 这里第一次操作目标值：交换目标值其右子树最左面节点。
        }
        root->left = deleteNode(root->left, key);
        root->right = deleteNode(root->right, key);
        return root;
    }
};
```
## 监控二叉树：贪心策略，叶子节点，开始看，然后给叶子结点的父节点安装，然后其父节点当作叶子节点继续重复

给定一个二叉树，在某些节点上安装摄像头，摄像头可以监控到该节点的父节点和子节点，求最少摄像头个数。

__因为需要从叶子节点开始，所以先递归遍历到叶子节点，所有机制应该在回溯的过程中进行。__





```c++
class Solution{
public:
    int num=0;
    int minCamera(TreeNode * & root){
        if(!root) return 0;
        int l=minCamera(root->left);
        int r=minCamera(root->right);
        if(!r&&!l) {return 1;}// 如果两个子节点都空，则返回1，代表当前节点是子节点
        else{ // 否则当前节点就不是子节点，那就应该放，并且放了之后当前这个节点就相当于空节点
            num++;
            return 0;
        }

    }
}

```


## 易错
### 最小二叉树深度: 这题递归思路和最大深度不一样，因为深度的定义是一定要算到叶子节点上，即如果只有一个儿子 minDepth(root.left,root.right)也不能为0

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        l=self.minDepth(root.left)
        r=self.minDepth(root.right)
        
        if l==0 or r==0:return l+1 if r==0 else r+1 
        return min(l,r)+1
```

### 平衡二叉树

单个递归函数版
```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def depth(root):
            if not root:return 0,True
            l,bl=depth(root.left)
            r,br=depth(root.right)
            if abs(l-r)>1 or not bl or not br:return 0,False 
            else: return max(l,r)+1,True
        _,result=depth(root)
        return result
```
拆成两个递归函数版： 感觉这样做不好啊，这样是不是开辟的栈空间更多了？
```c++
class Solution {
public:
    int deep(TreeNode *root){
        if(!root) return 0;
        return max(deep(root->left),deep(root->right))+1;
    }
    bool isBalanced(TreeNode* root) {
        if(!root) return true;
        return abs(deep(root->left)-deep(root->right))<=1&& isBalanced(root->left) && isBalanced(root->right);
    }
};
```

### 完全二叉树节点
```c++
 递归通用解法
class Solution {
public:
    int countNodes(TreeNode* root) {
        if(!root) return 0;
        return countNodes(root->left)+countNodes(root->right)+1;

    }
};

通用解法需要递归遍历所有节点，完全二叉树特定解法： 完全二叉树的节点数可以根据深度计算，而对于完全二叉树而言，只需要一直递归左子树到叶子就可以得到深度，因此不必遍历所有节点。

思路： 先得到左子树，右子树的深度，记为l,r,如果l==r，则左子树是一个满树，如果l>r,则右子树是个满树，满树直接用层数得到节点数，剩下的树的节点递归计算
class Solution {
public:
    int depth(TreeNode*root){
        int l=1;
        TreeNode* t=root;
        while(t){
            t=t->left;
            l++;
        }
        return l-1;
    }
    int countNodes(TreeNode* root) {
        if(!root) return 0;
        int l=depth(root->left);
        int r=depth(root->right);
        if(l==r) return countNodes(root->right)+(1<<l); 
        else return countNodes(root->left)+(1<<r);

    }
};
```

### 左下角节点的值：这个最好的应该是层序遍历，记录每层的开头，即队头的元素，但是双返回值递归也能做

递归的时候可以计算一下层数，如果右子树的左下角的值在的层数更深，那么就取右子树的，相等的时候再取左子树的值，和深度
```python

class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        def bottomleft(r):
            if not r: return 0,0
            if not r.left and not r.right:return r.val,1
            l,l_dep=bottomleft(r.left)
            r,r_dep=bottomleft(r.right)
            if r_dep>l_dep: return r,r_dep+1
            else:return l,l_dep+1
        result,_=bottomleft(root)
        return result
```

### 左叶子之和： 递归二叉树的时候可以在 两个子节点递归中插入操作

```c++
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if (root == NULL) return 0;

        int leftValue = sumOfLeftLeaves(root->left);    // 左
        if (root->left && !root->left->left && !root->left->right) { // 左子树就是一个左叶子的情况
            leftValue = root->left->val;
        }// 正常是没有这个if的，左子节点之和就应该等于左子树的左子节点和+右子树的左子节点和，但是如果当前遍历的这个点的左子树就是一个左字节点，那么直接修改其值就可以了，不然的话就不改变leftvalue
        int rightValue = sumOfLeftLeaves(root->right);  // 右

        int sum = leftValue + rightValue;               // 中
        return sum;
    }
};
```
### 二叉树的所有路径


dfs python版本：
```python
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        result=[]
        def dfs(r,path):
            if not r :return 
            path+=str(r.val)
            if not r.left and not r.right: result.append(path)
            if r.left: dfs(r.left,path+'->')
            if r.right:dfs(r.right,path+'->')
        dfs(root,"")
        return result
```

bfs python 版本：

```c++
// dfs 写法
class Solution {
public:
    void dfs(TreeNode* root,string path ,vector<string>&ans){
        if(!root) return;
        path+=to_string(root->val);
        if(!root->left&&!root->right) ans.push_back(path);
        else{
            path+="->";
            dfs(root->left,path,ans);
            dfs(root->right,path,ans);
        }
    }

    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> ans;
        dfs(root,"",ans);
        return ans;
    }
};

// BFS写法. 遍历好遍历，但是怎么存储那，BFS一行一行横着来，怎么才能记录一条向下的路径那？
// 用队列，队列能保证每次循环添加的一定是当前节点的儿子，然后把当前这个节点接上然后才放进队列就可以，这样遍历的时候就可以把路径记录上了，直到是空节点的时候，再把当前的队头存到结果列表里
class Solution {
public:
    
    vector<string> binaryTreePaths(TreeNode* root) {
        if(!root) return {""};
        queue<TreeNode*> tree;
        queue<string> s;
        vector<string> ans;
        
        s.push(to_string(root->val));
        tree.push(root);
        while(!tree.empty()){
            TreeNode * head=tree.front();
            string path=s.front();
            if(!head->left&&!head->right)ans.push_back(path);
            if(head->left) {tree.push(head->left);s.push(path+"->"+to_string(head->left->val));}
            if(head->right) {tree.push(head->right);s.push(path+"->"+to_string(head->right->val));}
            tree.pop();
            s.pop();
        }
        
        return ans;
    }
};
```

### 最大二叉树

```c++
class Solution {
public:
    TreeNode* build(int l,int r,vector<int>&nums){
        if(l>r) return nullptr;
        int max=nums[l];
        int id=l;
        for(int i=l;i<=r;i++)if(nums[i]>max){max=nums[i];id=i;}

        TreeNode * root=new TreeNode(nums[id]);
        root->left=build(l,id-1,nums);
        root->right=build(id+1,r,nums);
        return root;
    }

    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return build(0,nums.size()-1,nums);
    }
};
```

# 单调栈

单调栈用来求一个序列，每一位元素右边或左边第一个比他大或者小的元素的情况。

```c++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        stack<int> st; // 递增栈
        vector<int> result(T.size(), 0);
        for (int i = 0; i < T.size(); i++) {
            while (!st.empty() && T[i] > T[st.top()]) { // 注意栈不能为空
                result[st.top()] = i - st.top();
                st.pop();
            }
            st.push(i);

        }
        return result;
    }
};
```

```bash
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>
```