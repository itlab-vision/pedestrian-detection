function testCNN_ETH(out, Test_Boxes, Test_Frame, dstbasepaths)
    ImagSourcepath = '../../data/ETH/data-ETH';
    testFileName ='../../data/ETH/Params/ETHFileNames';
    scale = 2;

load(testFileName);

for a = 1:length(dstbasepaths)
    overlap = 0.6;

    model.Testing = true;
    VNeg = [];
    model.interval = 10;
    model.testPos = false;
    filnum = length(SrcNames);
    Scale_1 = 1/scale;
    PCNum = 1;

    for i = 1:PCNum:filnum
        jend = min(PCNum, filnum-i+1);
        for j=1:jend
            srcNames{j} = ImNames{i+j-1};
            dstbasepath = dstbasepaths{a};
            dstpath = [dstbasepath srcNames{j}(9:end-9)];
            if(~exist(dstpath,'dir')), mkdir(dstpath); end;
            dstName3{j} = [dstpath 'I' srcNames{j}(end-8:end)];
            dstName3{j}(end-2:end) = 'txt';
            dstmatNames{j}=dstName3{j};
            dstmatNames{j}(end-2:end)='mat';
        end;
        if i==1
            j = 1;
            srcName = srcNames{j};
            srcName2 = [ImagSourcepath srcName(1:end)];

        end
        j = 1;
            srcName = srcNames{j};
            srcName2 = [ImagSourcepath srcName(1:end)];

            box{j} = [Test_Boxes(Test_Frame.S(i):Test_Frame.E(i), 1:4) out{a}(1, Test_Frame.S(i):Test_Frame.E(i))'];

            if mod(i+j-1, 1000) == 0
                fprintf([': test: %d/%d %d found\n'],i+j-1,filnum, size(box{j}, 1));
            end;
            box2 = box{j};
            if ~isempty(box2)
                box2(:, 3:4) = box2(:, 3:4) - box2(:, 1:2) + 1;
            end
            box2 = nmsMy(box2, 0.5);
            if ~exist(dstName3{j}(1:end-10),'dir'),
                mkdir(dstName3{j}(1:end-10));
            end
            fid2 = fopen(dstName3{j}, 'w');
            if ~isempty(box2)

                boxsize = size(box2, 1);

                box2 = box2 .* Scale_1;

                if (boxsize>0)
                    fprintf(fid2, '%f, %f, %f, %f, %f\n', box2');
                end;
            end;
            fclose(fid2);
    end
end;

end

    
function top = nmsMy(boxes, overlap)

% top = nms(boxes, overlap) 
% Non-maximum suppression.
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected detection.

if isempty(boxes)
  top = [];
else
  x1 = boxes(:,1);
  y1 = boxes(:,2);
  x2 = boxes(:,3)+boxes(:,1);
  y2 = boxes(:,4)+boxes(:,2);
  s = boxes(:,end);
  area = (x2-x1+1) .* (y2-y1+1);

  [vals, I] = sort(s);
  pick = [];
  while ~isempty(I)
      supressLast = 0;
    last = length(I);
    i = I(last);
    s1 = vals(last);
    pick = [pick; i];
    suppress = [last];
    for pos = 1:last-1
      j = I(pos);
      s2 = vals(pos);
      xx1 = max(x1(i), x1(j));
      yy1 = max(y1(i), y1(j));
      xx2 = min(x2(i), x2(j));
      yy2 = min(y2(i), y2(j));
      w = xx2-xx1+1;
      h = yy2-yy1+1;
      if w > 0 && h > 0
        % compute overlap 
        o = w * h / area(j);
        o2 = w * h / area(i);

        %High first
        o = max(o2, o);
        
        %Big first
%         if (o2 > overlap) && (s2>0.5)
% %         if (o2 > overlap) && (s1>0) && (s2>0)
%             supressLast = 1;
%         end

        if o > overlap
          suppress = [suppress; pos];
        end
      end
    end
    if supressLast
        pick(end) = [];
    end;
    I(suppress) = [];
  end  
  top = boxes(pick,:);
end
end
