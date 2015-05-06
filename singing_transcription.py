# -*- coding: utf-8 -*-

__author__ = 'Roy'

from utilities import *
from utils import *
import scipy

"""
有一个很重要的问题
在所有的文献里面，似乎都默认这样一个事实：
受过训练的歌手的歌声里，一个note里面的pitch是稳定的，颤音都是稳的
未受训练的歌手在一个note内的pitch会晃来晃去，颤音也晃来晃去
但是实际上这个假设本身就不成立
由于艺术处理、唱法、咬字（尤其是咬字）、感情等的运用，受过训练的歌手的歌声也是会有非常大的波动的，甚至比未受训练的歌手更大
毕竟未受训练的歌手只会在note内部有问题，而受过训练的歌手的艺术处理可能出现在任何位置
几乎所有的文献拿来测试的数据集都是未受训练的歌声，或者受过训练的歌手唱的保持稳定的歌声
这种显然不可能推广到真实世界的情况
这样一来，例如连续100ms稳定的F0啊，pitch的剧烈变化啊这些分割note的方法都会有问题

并且真实世界中，一个音可能很短，不像我们想象的所有歌曲里一个音总能拖个接近1秒
我们用来测试的数据里就有很多几乎只有0.1s的pitch，但是我们仍然能听出来这里有一个音
如果一个pitch只有0.1s，那么用稳定0.1s的F0是不可能测得出来的了

另外一个问题是，alpha-trimmed weighting可能并不是确定note内我们感知到的音高的好方法
能量大的地方，也就是声音波形剧烈的地方，并不一定就是感知最强烈的地方
一个note内的波形可能有一些波动，某些波动大的地方可能pitch并不是我们感知到的，如果用能量加权的话就会出问题
这样的结果是，可能歌手唱的是准的，但是这样加权以后的pitch反而有偏差了
个人的感觉是人类能感知到的pitch时长很短，哪怕只有一下下也能感觉到，而忽略其他原因产生的pitch偏差
所以确定一个音的note的时候，一定不能用能量或别的什么加权
这一点在这段时间测试的过程里体会颇深

我的想法是，首先这个系统不是用来打分的，涉及到打分就怎么都要用主观的打分数据去训练，无法做到完全客观（absolutely objective）
所以这个系统的目的是，分析出哪些note不准，哪些地方节奏不对，把这些地方的时间点告诉用户，并告诉他音高了还是低了，节奏抢了还是拖了
算onset和offset还是可以按照连续的deviation来算，之后可以后处理
一定不能用加权，尤其能量非常不靠谱
确定note的时候我觉得就应该用histogram，哪个区域pitch最多就用哪个
这样能保证提取出来的pitch一定是感知的pitch
之后进行后处理来合并分的太细了的note
然后在装饰音检测的部分处理note中含有装饰音的情况，把装饰音单独标出来
颤音检测的部分对颤音进行标记，并且评价颤音的质量，用颤音频率和波动range来评价
然后评价的时候与标准去做对比，如果没有标准的话就分析note内的音高稳定度
有标准的话，对于提取出来的note序列去做节奏匹配和音高比较
节奏的时候需要把onset和offset全部归一到某一个比较整的单位，考虑到onset和offset可能不准，我觉得可以用0.2s左右作为单位，偏差在0.2以内都算准
音高方便一些，按照500cent为单位吧，半个半音差不多了
节奏的算法还是DTW，这个也想不出什么新的了

3.25更新：
按照之前的理论，一个音里面的note可能有很大的偏差，这个没有问题
但是两个音之间也可能只有很小的偏差，例如一个半音
这种放在上面就检测不出来（比如灌篮高手）
所以现在的总体思路是，把音高分割分为三个步骤
第一个步骤就像上面一样，以大的threshold分割，原理是一个note内偏差可能很大
第二个步骤还是合并，跟上面一样
这两步保证的是不会多检，其实就是求可能的note数量的最小值的过程
第三个步骤是对于那些很长的note，用一个小的threshold检查里面是不是有更细的note
这里只检查具有稳定音高且音高不同于大的note pitch的小note，只可能在一个semitone左右的范围内
可以采用类似于某文献的持续100ms（10个点）的方法
这一步保证的是不会漏检，把之前粗粒度的检测细化
然后再进行装饰音颤音等等的检测
这样精度可以进一步提升，我估计这样的精度已经可以很高了
并且的确适用于各种类型的旋律走向

4.11更新
颤音要放在小threshold分割之前，避免把过慢的颤音给分割了
加入了一个onset/offset调整的步骤，处理分割的不准的onset和offset
思路是把onset和offset（只讨论连续的note，即onset和offset差距不超过3）放在note中间
检测前一个note的最后一个属于histogram bin的pitch和后一个note的第一个属于histogram bin的pitch
把新的onset和offset设为这两个的中点
"""




# 初步分割note
# 计算累积cent偏差（100cent为一个半音，设一个200左右的阈值），初始值为0
# 如果有连续的若干个点满足maximum deviation不超过某个threshold，则为一个note
# 也就是说，使用一个矩形窗，计算每个窗内的min, max和maximum deviation
# 一旦累积超过threshold，清零
# 如果没有超过，则这个窗的开头为onset candidate，从这20个点之后的第一个点逐个扫描，直到deviation+窗里的deviation超过阈值
# 这样确定的onset一定不会在无声区域
# 超过处为offset candidate，从这里从新开始用窗扫描
# 这样的offset也肯定不在无声区域
# 最后用histogram算目前的每个note内的pitch
# 这个方法可以避免滑音被分成note，同时防止note被漏检（比如某算法中的用连续100ms稳定F0来算note，会漏很多note）
# 可能会把note分的过细，比如把装饰音单独分出去了，所以需要后处理
# 也可能会把note分的过粗，因为这里阈值用的很高，是为了避免单个音内部波动很大
# 但是对于某些音高来回在两个相邻半音上变化的乐句（比如灌篮高手），这个就分不出来了
# 所以要进行两步后处理，第一步先合并过细的，第二步检测内部有没有连续的半音变化
# 合并的时候分情况讨论
# 检测内部的半音的时候，在note内寻找是否有异于note pitch的稳定（至少10个点）pitch，且这个稳定pitch跟note pitch差值在85cent以上
# 若有，那么把这一块稳定区域单独拿出来作为一个新note

def note_segment(time, pitch):
    # time为X轴，pitch为Y轴
    threshold = 170.
    window_size = 15
    onset_candidate = []
    offset_candidate = []

    num = 0
    while num < len(time) - window_size - 1:
        # num为序号
        cumuative_deviation = 0.
        range_max = 0.
        range_min = 0.

        for i in range(num, num + window_size):  # 加窗，计算deviation
            if i == num:
                range_max = pitch[i]
                range_min = pitch[i]
                cumuative_deviation = 0.
            else:
                if pitch[i] > range_max:
                    cumuative_deviation = max(cumuative_deviation, pitch[i] - range_min)
                    range_max = pitch[i]
                elif pitch[i] < range_min:
                    cumuative_deviation = max(cumuative_deviation, range_max - pitch[i])
                    range_min = pitch[i]

        if cumuative_deviation > threshold or min(pitch[num: num + window_size]) < 500:  # 超过threshold或有点在无声区
            num += 1
            continue
        else:
            onset_candidate.append(num)  # onset candidate为窗的第一个时间值
            # 之后的点逐一扫描
            num += window_size
            while cumuative_deviation <= threshold and num < len(time)-1:
                if pitch[num] > range_max:
                    cumuative_deviation = max(cumuative_deviation, pitch[num] - range_min)
                    range_max = pitch[num]
                elif pitch[num] < range_min:
                    cumuative_deviation = max(cumuative_deviation, range_max - pitch[num])
                    range_min = pitch[num]
                num += 1
            offset_candidate.append(num-2)  # 超过时，该点为offset candidate
            num -= 1

    # note以[onset，offset，pitch]的三元组形式存储，为一个嵌套list
    note = []
    for i in range(len(onset_candidate)):
        note.append([onset_candidate[i], offset_candidate[i], histogram_mean(onset_candidate[i], offset_candidate[i], pitch)])
    return note


# 对上面初步提取的note candidate后处理
# 处理note的合并，有两种情况会合并
# 由于有上面的maximum deviation的限制，这样合并出来的note的pitch一定不会有大的波动
# 然后更新新的note里面的pitch，作为最终的输出

def note_postprocessing(note, full_pitch):

    # 处理note的合并
    # 要看onset和offset的距离

    cnt = 0
    length = len(note)
    while cnt < length - 1:
        delete_size = int((note[cnt][1] - note[cnt][0]) / 20)
        #  前一个的offset跟后一个的onset距离不超过5
        if note[cnt+1][0] - note[cnt][1] <= 3:
            # 有两种情况会合并
            # 1. 如果pitch之差小于45，合并
            # 2. 某一个note内pitch只有一个峰值（就是正着的”V“和倒着的“V“）且note pitch与所有pitch的均值差别大于40（避免出现V的底端很宽的情况），一定合并

            # 先处理第一种
            if abs(note[cnt+1][2] - note[cnt][2]) < 45:
                note[cnt] = [note[cnt][0], note[cnt+1][1], histogram_mean(note[cnt][0], note[cnt+1][1], full_pitch)]
                del note[cnt+1]
                length -= 1

            # 再处理第二种
            elif abs(note[cnt][2] - np.mean(full_pitch[note[cnt][0]:note[cnt][1]+1])) > 40:
                # 在note内计数一下峰值
                # 只数前面的note
                # 双侧扫描，以免出现false peak
                current_pitch = full_pitch[note[cnt][0] + delete_size: note[cnt][1]-delete_size]
                left = 0
                right = len(current_pitch)-1
                left_min = current_pitch[0]
                left_max = current_pitch[0]
                right_min = current_pitch[-1]
                right_max = current_pitch[-1]
                max_pos = 0
                min_pos = 0
                while left < right:
                    # 先看左边
                    if current_pitch[left] < left_min and current_pitch[left] < right_min:
                        # 为当前最佳波谷
                        left_min = current_pitch[left]
                        min_pos = left
                    if current_pitch[left] > left_max and current_pitch[left] > right_max:
                        # 为当前最佳波峰
                        left_max = current_pitch[left]
                        max_pos = right
                    # 再看右边
                    if current_pitch[right] < left_min and current_pitch[right] < right_min:
                        # 为当前最佳波谷
                        min_pos = right
                        right_min = current_pitch[right]
                    if current_pitch[right] > left_max and current_pitch[right] > right_max:
                        # 为当前最佳波峰
                        max_pos = right
                        right_max = current_pitch[right]
                    left += 1
                    right -= 1
                if min_pos * max_pos == 0:
                    if min_pos + max_pos == 0:
                        # 表示没有波峰和波谷，是单调的
                        cnt += 1
                        continue
                    else:
                        # 表示只有一个波峰或波谷，先看它的波峰或波谷是不是false的
                        # 通过跟两个端点的pitch比较，差值都要大于50
                        # 满足，则跟它前后两个note中pitch更接近的一个合并
                        # 比较前后note跟它的pitch
                        # 如果它是第一个或最后一个，那么不用看了直接合并
                        if min(abs(current_pitch[max(min_pos, max_pos)] - current_pitch[0]), abs(current_pitch[max(min_pos, max_pos)] - current_pitch[-1])) < 50:
                            cnt += 1
                            continue
                        else:
                            if cnt - 1 <= 0 or cnt + 1 >= length - 1:
                                note[cnt] = [note[cnt][0], note[cnt+1][1], histogram_mean(note[cnt][0], note[cnt+1][1], full_pitch)]
                                del note[cnt+1]
                                length -= 1
                            else:
                                # 如果是中间的，那么比较pitch
                                if abs(note[cnt-1][2] - note[cnt][2]) < abs(note[cnt][2] - note[cnt+1][2]) and note[cnt][0] - note[cnt-1][1] <= 5:
                                    # 跟前面的合并，注意这时候合并的是cnt-1，这里处理完之后cnt是不能++的
                                    note[cnt-1] = [note[cnt-1][0], note[cnt][1], histogram_mean(note[cnt-1][0], note[cnt][1], full_pitch)]
                                    del note[cnt]
                                    length -= 1
                                    cnt -= 1
                                else:
                                    # 跟后面的合并
                                    note[cnt] = [note[cnt][0], note[cnt+1][1], histogram_mean(note[cnt][0], note[cnt+1][1], full_pitch)]
                                    del note[cnt+1]
                                    length -= 1
        cnt += 1

    return note


# 细粒度note分割
# 这玩意很麻烦，尝试了三种方法了都不好
# 这一步的目的是分割由于上面粗粒度分割和合并后，可能没有检测出来的note
# 这种note的音高跟上面的一个大note的音高差都在一个semitone左右
# 思路是在大note里面，寻找稳定且具有一定长度的音高点，长度可以设为100ms（10个点）
# 这个基本就是某文献里面那个稳定100ms的思路了
# 然后把大note切成一个一个小note
# 并且要求新note的pitch跟原来的note pitch相比要相差至少80cent
# 如果切出来的还很长就继续检测里面
# 大概就是一个BFS的思想
# 在检测的时候，只检测那些在算histogram的时候不在note pitch所在的区间内，离那个bin的边界的距离大于30的bin里面的那些点
# 不然也把接近note pitch的点算进去会非常麻烦
# 建立一个bool数组，存放每个点究竟满不满足上面这个条件
# 只在满足条件的点里面找连续的，不然又会切的太细
# 这样子可以保证颤音不会被切，因为颤音里不可能有这样连续的一篇比note pitch高不少的区域


def small_note_segment(note, full_pitch, vibrato):
    # 寻找时长大于35个点的note
    # 选35是因为35的长度是grace note的可能最大长度，这样的note一定要比grace note长
    pos = 0
    length = len(note)
    while pos < length:
        single_note = note[pos]
        if single_note[1] - single_note[0] > 35:
            # 建立bool数组，寻找那些pitch跟note pitch相差在65以上的点
            current_note = full_pitch[single_note[0]:single_note[1]+1]
            equal_to_note_pitch = [True] * len(current_note)
            for i in range(len(current_note)):
                if abs(current_note[i] - single_note[2]) > 65.:
                    equal_to_note_pitch[i] = False
            # 只在bool数组为True的点里面找所有连续的区域
            # 如果有的话，看这段连续的区域里面的pitch偏差是不是在65cent以内
            # 这样切分的好处是，如果在pitch里面有很大的波动，那么会被切开，这样在跟标准做对比的时候就会发现错误
            # 所以说这是一种适合于歌唱评价的transcription方法
            max_pitch = 0
            min_pitch = 0
            pitch_region = []
            for i in range(len(current_note)):
                # 先看是不是不等于note pitch
                if not equal_to_note_pitch[i]:
                    # 是不是当前区域的第一个
                    if len(pitch_region) == 0:
                        pitch_region.append(i)
                        max_pitch = current_note[i]
                        min_pitch = current_note[i]
                    else:
                        # region非空，已经不止一个了
                        # 看max-min的范围是不是在65以内
                        if current_note[i] > max_pitch and current_note[i] - min_pitch < 65.:
                            pitch_region.append(i)
                            max_pitch = current_note[i]
                        elif current_note[i] < min_pitch and max_pitch - current_note[i] < 65.:
                            pitch_region.append(i)
                            min_pitch = current_note[i]
                        elif min_pitch < current_note[i] < max_pitch:
                            pitch_region.append(i)
                        else:
                            # 范围超过了65，判断这段区域是否要单独分割出来
                            # 先看这段区域有多长，如果大于15个点（选15是因为修饰音范围为15），需要分割，不然一定不分割
                            if len(pitch_region) < 15:
                                pitch_region = []
                                max_pitch = 0
                                min_pitch = 0
                                i = i - len(pitch_region) + 2
                            else:
                                # 长度够了，根据这段区域所在的位置判断是切成两份还是三份
                                note, pitch_region, pos, length = split_note(note, pitch_region, single_note, pos, length, full_pitch, vibrato)
                                break
                else:
                    # 此时这个点不符合要求了，看至今为止的片段是否能切出来
                    if len(pitch_region) < 15:
                        pitch_region = []
                        max_pitch = 0
                        min_pitch = 0
                    else:
                        # 长度够了，根据这段区域所在的位置判断是切成两份还是三份
                        note, pitch_region, pos, length = split_note(note, pitch_region, single_note, pos, length, full_pitch, vibrato)
                        break
            # 如果扫到底了并且最后这一段足够长，切成两份
            if len(pitch_region) > 15:
                length += 1
                del note[pos]
                note.insert(pos, [single_note[0]+pitch_region[0], single_note[1], histogram_mean(single_note[0]+pitch_region[0], single_note[1], full_pitch)])
                note.insert(pos, [single_note[0], single_note[0]+pitch_region[0]-2, histogram_mean(single_note[0], single_note[0]+pitch_region[0]-2, full_pitch)])
                pos += 1
            else:
                # 扫描下一个
                pos += 1
        else:
            # 扫描下一个note
            pos += 1
    return note

# 调整onset和offset的位置
# 由于threshold的选取可能导致一些onset和offset没对齐
# 这种情况挺常见的，尤其是在pitch变化快的区域
# 调整的思路是，对相邻的两个note，检查前一个note的最后一个和后一个note的第一个处于pitch note histogram的frame
# 然后把onset更新为这两个frame的中点
# 这个过程只对相邻onset和offset差距不超过3的note进行检测，也就是pitch连续变化的区域

def onset_offset_adjust(note, full_pitch):
    for cnt in range(len(note)-1):
        current_note = note[cnt]
        next_note = note[cnt+1]
        if next_note[0] - current_note[1] <= 3 and min(full_pitch[current_note[1]:next_note[0]+1]) > 0:
            # 检测处于histogram的区域
            # 先检测当前note
            current_note_pitch = full_pitch[current_note[0]:current_note[1]+1]
            current_pos = 0
            for i in range(len(current_note_pitch)-1, -1, -1):
                if abs(current_note_pitch[i] - current_note[2]) >= 15.:
                    continue
                else:
                    current_pos = i
                    break
            # 然后检测后面的note
            next_note_pitch = full_pitch[next_note[0]:next_note[1]+1]
            next_pos = 0
            for i in range(len(next_note_pitch)):
                if abs(next_note_pitch[i] - next_note[2]) >= 15.:
                    continue
                else:
                    next_pos = i
                    break
            # 考虑到如果后面note的开头有修饰音，那么第一个处于histogram内的点可能很远
            # 这样可能有问题
            # 所以如果这个点太远了，那么就不调整了，继续往下扫描
            if next_pos > 15:
                continue
            # 调整onset和offset为这两个位置的中点
            middle = int((current_note[0]+current_pos + next_note[0]+next_pos)/2)
            note[cnt][1] = middle-1
            note[cnt+1][0] = middle+1
    return note


# 检测修饰音，修饰音只可能出现在被修饰音的前面，并且一般为低小于等于一个半音，时长为被修饰音的一半以下
# 检查每一对相邻的note的pitch，如果offset和onset差距不超过3、pitch差距小于200（一个全音）且时长小于min（25, 后一个音的一半），则标记为修饰音
# 然后检查单个的note，单个的note内也有可能有小于threshold的修饰音
# 在单个的note内从左往右扫描，如果在pitch小于note pitch的区域内只有一个波谷，比较它的pitch和整个note的pitch
# 如果差值大于100cent，则以这个波谷为中心向两侧扫描直到遇到onset，这时候划分出来的区域确定为修饰音，标记出来
# 标记修饰音的意义在于，在进行最终的对比评价的时候，可以提醒用户这里有修饰音，可以选择唱或不唱
# 如果修饰音在一个note内，那么不影响这个note的音高，所以不影响评价
# 如果修饰音是单独的一个note，那么评价的时候要把这个note不考虑在内，唱不唱修饰音都不算错

def grace_note_detection(note, full_pitch):
    grace_note = []
    # 先检查每一对相邻的note，相邻note的offset和onset相差不超过3
    # 如果前面的note时长小于15且音高差在220cent以内，算
    # 如果大于15小于min(25, 后一个音的一半)且音高差在220以内，也算
    for i in range(0, len(note)-1):
        if note[i+1][0] - note[i][1] <= 3 and 10 < note[i][1] - note[i][0] < 15 and 70 < note[i+1][2] - note[i][2] < 220:
            grace_note.append(note[i])
        elif note[i+1][0] - note[i][1] <= 3 and 70 < note[i+1][2] - note[i][2] < 220 and 15 < note[i][1] - note[i][0] < min((note[i+1][1] - note[i+1][0])/2, 25) < 100:
            grace_note.append(note[i])
    # 然后检查单个的note
    for single_note in note:
        current_pitch = full_pitch[single_note[0]:single_note[1]+1]
        # 从左往右扫描，在pitch小于note的pitch的时候数波谷和波峰，找出第一个波峰和波谷，要求他们都小于note pitch
        # 同样去除最左的5%
        peak = []
        pitch_min = current_pitch[len(current_pitch)/20]
        pitch_max = 0
        pitch_max_pos = 0
        for i in range(len(current_pitch)-1, 5, -1):
            # 找到从右到左数最后一个等于pitch note的点
            if single_note[2] - 15 < current_pitch[i] < single_note[2] + 15:  # 30为histogram的bin的大小
                pitch_max = current_pitch[i]
                pitch_max_pos = i
        bottom = 0
        top = 0
        for i in range(len(current_pitch)/20, pitch_max_pos):
            if single_note[2] > pitch_min > current_pitch[i]:
                bottom = i + single_note[0]
                pitch_min = current_pitch[i]
            elif pitch_max < current_pitch[i] < single_note[2]:
                top = i + single_note[0]
                pitch_max = current_pitch[i]
            elif current_pitch[i] > single_note[2]:
                break
        if min(bottom, top) > 0:
            # 有两个
            peak.append(min(bottom, top))
            peak.append(max(bottom, top))
        elif max(bottom, top) > 0:
            # 有一个
            peak.append(max(bottom, top))
        else:
            continue
        if 1 <= len(peak) <= 2:
            if single_note[2] - histogram_mean(single_note[0], single_note[0] + pitch_max_pos, full_pitch) > 90 and pitch_max_pos > 10:
                # onset到刚好小于pitch note的位置的区域为修饰音
                # 修饰音音高仍然用过histogram确定
                grace_note.append([single_note[0], single_note[0] + int(0.9 * pitch_max_pos), histogram_mean(single_note[0], single_note[0] + int(0.9 * pitch_max_pos), full_pitch)])

    return grace_note



# 检测颤音，颤音可能在任何时间点，而且不一定在note的结尾，note的开头都有可能，所以要对每个note检测
# 这个方法尝试了很久，由于颤音内部本身不稳定，所以纯粹从pitch的角度还是很难处理的

def vibrato_detection(note, full_pitch):
    vibrato = []

    for note_num in range(len(note)):
        single_note = note[note_num]

        # 长度太短的note不考虑
        if single_note[1] - single_note[0] < 50:
            continue
        current_pitch = full_pitch[single_note[0]:single_note[1]+1]

        # 先把每个点的pitch都减掉这个note的值
        for i in range(len(current_pitch)):
            current_pitch[i] -= single_note[2]

        # 然后寻找所有pitch大于30cent小于150cent的区域
        # 选30和150的原因是文献里说vibrato的范围可能为30-150cent
        vibrato_possible = [False] * len(current_pitch)
        for i in range(len(current_pitch)):
            if 30. < abs(current_pitch[i]) < 150.:
                vibrato_possible[i] = True

        # 然后使用自相关来寻找当前note里面的周期性
        # 这个算法完全就是YIN的算法了
        # 在满足4-9Hz周期性的点里面，看是否满足上面的条件
        # 如果满足，且这一段周期的区域里至少有4个峰值，那么认为这个note里面包含颤音

        # 将lag设为1-30，求整段pitch的自相关函数值
        # 这里只算满足上面条件（30-150cent区域内）的点的自相关
        # 其他点的pitch都算成0
        for i in range(len(current_pitch)):
            if not vibrato_possible:
                current_pitch[i] = 0

        acf_diff = []
        for lag in range(15, len(current_pitch)/2-1):
            temp = 0.
            for i in range(0, len(current_pitch)/2):
                temp += np.power((current_pitch[i] - current_pitch[i+lag]), 2)
            acf_diff.append(temp)

        # 应该选出最小的值对应的lag
        # 前提是此时的acf值小于阈值
        # 这里取阈值为15w
        if min(acf_diff) < 200000:
            # 这一段可能是颤音区或平稳区，要判断一下
            possible_lag = 15 + acf_diff.index(min(acf_diff))
            # 找出所有的极大值，看一下是否存在某些极大值的子序列满足相差均在lag左右，找到3个就算颤音了

            # 找波峰
            peak = []
            for i in range(1, len(current_pitch)-1):
                if current_pitch[i-1] < current_pitch[i] > current_pitch[i+1]:
                    peak.append(i)

            # 寻找peak中是否存在某个子序列的公差在所求的周期左右（相差5以内）
            # 采用方法是从头开始扫描，看每个元素+lag后的位置有没有新的peak，没有就重新扫描
            pos = 0
            length = len(peak)
            vibrato_peak = []
            while pos < length:
                if len(vibrato_peak) == 0:
                    vibrato_peak.append(peak[pos])
                else:
                    flag = False
                    for deviation in range(-5, 6):
                        if vibrato_peak[-1] + possible_lag + deviation in peak:
                            vibrato_peak.append(vibrato_peak[-1] + possible_lag + deviation)
                            flag = True
                            break
                    if not flag:
                        # 没了，看长度有没有超过3
                        if len(vibrato_peak) >= 3:
                            # 看这个peak跟前一个peak相差是否不大于35cent
                            for peaks in range(1, len(vibrato_peak)):
                                if abs(current_pitch[vibrato_peak[peaks-1]] - current_pitch[vibrato_peak[peaks]]) > 50.:
                                    # 这个peak不符合条件了，看至今为止的区域有没超过3个peak
                                    if peaks-1 >= 2:
                                        # 是颤音
                                        vibrato.append([single_note[0] + vibrato_peak[0], single_note[0] + vibrato_peak[peaks]])
                                        # 重新算pitch，设为整片的均值
                                        note[note_num][2] = np.mean(full_pitch[single_note[0] + vibrato_peak[0]:single_note[0] + vibrato_peak[peaks]+1])
                                        vibrato_peak = []
                                        break
                                    else:
                                        # 以pos开头的这个peak区域不是颤音，继续搜索之后的peak区域
                                        pos += 1
                                        vibrato_peak = []
                                        break

                            # 看搜到最后的时候是否还能构成一个颤音
                            if len(vibrato_peak) >= 3:
                                # 是颤音
                                vibrato.append([single_note[0] + vibrato_peak[0], single_note[0] + vibrato_peak[peaks]])
                                # 重新算pitch，设为整片的均值
                                note[note_num][2] = np.mean(full_pitch[single_note[0] + vibrato_peak[0]:single_note[0] + vibrato_peak[peaks]+1])
                                vibrato_peak = []

                        # 如果长度不到3，以pos开头的这一段不是颤音，重新换pos搜索
                        else:
                            pos += 1
                            vibrato_peak = []

                    # 如果找到了这个距离的peak，继续往后找
                    else:
                        continue

                # 看是不是检测出颤音了，是的话就不用继续了
                if len(vibrato) > 0 and vibrato[-1][0] > single_note[0]:
                    break
    return vibrato, note




# 计算histogram
# 不用alpha-trimmed weighting的原因在上面讲了
# 把note内分为5个区域，看哪个区域的点最多
# 需要首先剔除首尾时间各5%，剔除一下边界的pitch outlier
# 因为在我们的规定下note内不会有超过一个全音的偏差，那么outlier肯定在边界

def histogram_mean(onset_candidate, offset_candidate, full_pitch):
    # 只选取总时间的90%，剔除首尾各5%
    total_length = offset_candidate - onset_candidate + 1
    current_pitch = full_pitch[onset_candidate + int(total_length * 0.05):offset_candidate + 1 - int(total_length * 0.05)]
    min_pitch = min(current_pitch)
    max_pitch = max(current_pitch)
    if max_pitch - min_pitch < 30:
        return np.mean(current_pitch)
    range_size = (max_pitch - min_pitch) / 5.
    # 分五个区间
    pitch_range0 = 0
    pitch_range1 = 0
    pitch_range2 = 0
    pitch_range3 = 0
    pitch_range4 = 0
    # 看谁的pitch最多
    for pitch in current_pitch:
        block = int((pitch - min_pitch) / range_size)
        if block == 0:
            pitch_range0 += 1
        elif block == 1:
            pitch_range1 += 1
        elif block == 2:
            pitch_range2 += 1
        elif block == 3:
            pitch_range3 += 1
        elif block == 4:
            pitch_range4 += 1
        elif block == 5:
            pitch_range4 += 1
    histogram = [pitch_range0, pitch_range1, pitch_range2, pitch_range3, pitch_range4]
    pos = histogram.index(max(histogram))
    weighted_pitch = np.mean([pitch for pitch in current_pitch if min_pitch + range_size * pos < pitch < min_pitch + range_size * (pos + 1)])
    return weighted_pitch


# 判断是否要分割一个note
# 并确定怎么分割

def split_note(note, pitch_region, single_note, pos, length, full_pitch, vibrato):
    # 首先看要分割的pitch_region在不在某个vibrato里面，在的话不能分割
    for i in range(len(vibrato)):
        if vibrato[i][0] <= single_note[0] + pitch_region[0] < single_note[0] + pitch_region[-1] <= vibrato[i][1]:
            # 不分割
            pos += 1
            pitch_region = []
            return note, pitch_region, pos, length
    # 如果在onset附近，且若切成两份时后一份（新note的offset到原始note的offset）的pitch等于note pitch，切成两份
    if pitch_region[0] < 15 < single_note[1] - (single_note[0] + pitch_region[-1]):
        if abs(histogram_mean(single_note[0] + pitch_region[-1], single_note[1], full_pitch) - single_note[2]) < 45.:
            length += 1
            del note[pos]
            note.insert(pos, [single_note[0]+pitch_region[-1], single_note[1], histogram_mean(single_note[0]+pitch_region[-1], single_note[1], full_pitch)])
            note.insert(pos, [single_note[0], single_note[0]+pitch_region[-1]-2, histogram_mean(single_note[0], single_note[0]+pitch_region[-1]-2, full_pitch)])
            pos += 1
            pitch_region = []
        # 如果后面的pitch不等于note pitch，不分割
        else:
            pos += 1
            pitch_region = []

    # 然后如果在offset附近，且切成两份时原始note的onset到新note的onset那一片的pitch等于note pitch，切成两份
    elif single_note[1] - (single_note[0] + pitch_region[-1]) < 15 < pitch_region[0]:
        if abs(histogram_mean(single_note[0], single_note[0] + pitch_region[0], full_pitch) - single_note[2]) < 45.:
            length += 1
            del note[pos]
            note.insert(pos, [single_note[0]+pitch_region[0], single_note[1], histogram_mean(single_note[0]+pitch_region[0], single_note[1], full_pitch)])
            note.insert(pos, [single_note[0], single_note[0]+pitch_region[0]-2, histogram_mean(single_note[0], single_note[0]+pitch_region[0]-2, full_pitch)])
            pos += 1
            pitch_region = []
        # 如果不等于，不分割
        else:
            pos += 1
            pitch_region = []

    # 如果在中间，肯定切成三段
    else:
        length += 2
        del note[pos]
        note.insert(pos, [single_note[0]+pitch_region[-1]+2, single_note[1], histogram_mean(single_note[0]+pitch_region[-1]+2, single_note[1], full_pitch)])
        note.insert(pos, [single_note[0]+pitch_region[0], single_note[0]+pitch_region[-1], histogram_mean(single_note[0]+pitch_region[0], single_note[0]+pitch_region[-1], full_pitch)])
        note.insert(pos, [single_note[0], single_note[0]+pitch_region[0]-2, histogram_mean(single_note[0], single_note[0]+pitch_region[0]-2, full_pitch)])
        pos += 1
        pitch_region = []

    return note, pitch_region, pos, length