const tableInit=document.querySelector("#table0");
tableInit.classList.remove("hidden");
tableInit.classList.add("table");

const allTable=document.querySelectorAll(".table");
count=0
const prev=document.querySelector("#prev");
const nxt=document.querySelector("#next");

prev.addEventListener("click",()=>{
if(count>0)
{
    allTable[count].classList.add("hidden");
    allTable[count].classList.remove("table"); 
    count-=1
    allTable[count].classList.remove("hidden");
    allTable[count].classList.add("table");
}
});

next.addEventListener("click",()=>{
    if(count<allTable.length-1)
    {
        allTable[count].classList.add("hidden");
        allTable[count].classList.remove("table"); 
        count+=1
        allTable[count].classList.remove("hidden");
        allTable[count].classList.add("table");
    }
    });
