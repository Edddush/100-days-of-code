# 100 Days Of Code - Log

```mermaid
journey
    title MY 100-DAY CODING JOURNEY
    section LC-1768
      D001: 9: Arrays
      D002: 6: Arrays, GCD
      D---: 9: Sets
      D---: 9: Stack
      D---: 9: Queue
      D---: 9: Tree
```
> [!IMPORTANT]
> All solutions from this log are my own
> I am keeping track of my progress as I go through the challenge
> I hope to see constant improvements on performance, algorithms, and data structures choices

## Day 1
Public accountability and date on [X](https://x.com/Edddushi/status/1799998093464838351)
### Accomplishments
- Solved the Merge Strings Altenately on Leetcode #1768
- Learned how to traverse an array with multiple indices
### Challenges
- Struggled understanding all moving parts at first, I started actually thinking out loud at the end
  
<details>
  <summary>CLICK TO VIEW SOLUTION</summary>

   ```python
    def mergeAlternately(self, word1: str, word2: str) -> str:
        len1 = len(word1)
        len2 = len(word2)
        size = min(len1, len2) * 2
        result = ""
        counter = 0
        i = 0
        j = 0

        while size > 0:
            if counter % 2 == 0:
                result += word1[i]
                i += 1
            else:
                result += word2[j]
                j += 1
            size -= 1
            counter += 1

        if len1 > len2:
            result += word1[i:]
        elif len1 < len2:
            result += word2[j:]

        return result
   ```
  
</details>


# D002
Public accountability and date on [X](https://x.com/Edddushi/status/1800370219149656207)

### Accomplishments
- Solved the Greatest Common Divisor of Strings on Leetcode #1071
- In hindisght I should have been able to tell by the name I would need my Maths hat for this one, but I learnt it the hard way
### Challenges
- This felt like an easy hard question. Not because the solution is particularly hard to implement, realizing the maths behind it is the tricky part
  
<details>
  <summary>CLICK TO VIEW SOLUTION</summary>

   ```python
    class Solution:
        def gcdOfStrings(self, str1: str, str2: str) -> str:
            common_chars = []
            size1 = len(str1)
            size2 = len(str2)
            longer_word = str1 if (size1 > size2) else str2
            shorter_word = str2 if (size1 > size2) else str1
            common_str = shorter_word
            i = 1
    
            if( size1 == 0 or size2 == 0):
                return ""
    
            for j in range(min(size1, size2)):
                gcd = len(common_str)
    
                if(size1 % gcd == 0 and size2 % gcd == 0):
    
                    multiplier1 = int(size1 / len(common_str))
                    multiplier2 = int(size2 / len(common_str))
    
                    if(str1 == common_str * multiplier1 and str2 == common_str * multiplier2):
                        return common_str
    
                common_str = shorter_word[:-i]
                i += 1 
    
            return ""
   ```
  
</details>


<!---
### Day 0: February 30, 2016 (Example 1)
##### (delete me or comment me out)

**Today's Progress**: Fixed CSS, worked on canvas functionality for the app.

**Thoughts:** I really struggled with CSS, but, overall, I feel like I am slowly getting better at it. Canvas is still new for me, but I managed to figure out some basic functionality.

**Link to work:** [Calculator App](http://www.example.com)

### Day 0: February 30, 2016 (Example 2)
##### (delete me or comment me out)

**Today's Progress**: Fixed CSS, worked on canvas functionality for the app.

**Thoughts**: I really struggled with CSS, but, overall, I feel like I am slowly getting better at it. Canvas is still new for me, but I managed to figure out some basic functionality.

**Link(s) to work**: [Calculator App](http://www.example.com)


### Day 1: June 27, Monday

**Today's Progress**: I've gone through many exercises on FreeCodeCamp.

**Thoughts** I've recently started coding, and it's a great feeling when I finally solve an algorithm challenge after a lot of attempts and hours spent.

**Link(s) to work**
1. [Find the Longest Word in a String](https://www.freecodecamp.com/challenges/find-the-longest-word-in-a-string)
2. [Title Case a Sentence](https://www.freecodecamp.com/challenges/title-case-a-sentence)
--->
